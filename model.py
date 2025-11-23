import math
import torch
from torch import nn
import torch.nn.functional as F
import vs_helper


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)
        return y, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, num_feature=1024, dropout=0.5):
        super().__init__()
        self.num_head = num_head
        self.Q = nn.Linear(num_feature, num_feature, bias=False)
        self.K = nn.Linear(num_feature, num_feature, bias=False)
        self.V = nn.Linear(num_feature, num_feature, bias=False)
        self.d_k = num_feature // num_head
        self.attention = ScaledDotProductAttention(self.d_k, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(num_feature, num_feature, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        _, seq_len, num_feature = x.shape

        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        K = K.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        Q = Q.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)
        V = V.view(1, seq_len, self.num_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(self.num_head, seq_len, self.d_k)

        y, attn = self.attention(Q, K, V)
        y = y.view(1, self.num_head, seq_len, self.d_k).permute(0, 2, 1, 3).contiguous().view(1, seq_len, num_feature)

        y = self.fc(y)
        return y, attn


class AttentionExtractor(MultiHeadAttention):
    def forward(self, *inputs):
        out, _ = super().forward(*inputs)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Reconstruction(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.fc1 = nn.Linear(num_feature, num_feature * 2)
        self.fc2 = nn.Linear(num_feature * 2, num_feature)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, feature):
        hidden = self.lrelu(self.fc1(feature))
        out = self.fc2(hidden)
        return self.sigmoid(out)


class STeMI(nn.Module):
    def __init__(self, num_feature, num_hidden, num_head, temporal_scales, spatial_scales, dropout=0.5):
        super().__init__()

        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.temporal_scales = temporal_scales
        self.spatial_scales = spatial_scales

        self.attention = AttentionExtractor(num_head, num_feature, dropout)
        self.spatial_fc_1 = nn.Linear(num_feature, num_feature)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 1, 32))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 1, 32))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 1, 32))
        nn.init.trunc_normal_(self.pos_embed_1, std=.02)
        nn.init.trunc_normal_(self.pos_embed_2, std=.02)
        nn.init.trunc_normal_(self.pos_embed_3, std=.02)

        # learnable weights
        self.temporal_scale_weights = nn.Parameter(torch.ones(temporal_scales))
        self.spatial_scale_weights = nn.Parameter(torch.ones(spatial_scales))

        self.reconstruction = Reconstruction(num_feature)

        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(num_hidden)
        )

        self.merge_extractor = nn.Sequential(
            nn.Linear(num_feature, num_feature),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(num_feature)
        )

        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)

    def forward(self, x, support_feature, support_target):
        support_target = support_target.squeeze(0)
        support_summary = support_feature[:, support_target, :]

        # apply positional embedding
        spatial_support_feature = self.spatial_fc_1(
            support_feature + self.pos_embed_1.repeat(1, 32, 1).transpose(1, 2).reshape(1, 1, -1)
        )
        spatial_support_summary = self.spatial_fc_1(
            support_summary + self.pos_embed_2.repeat(1, 32, 1).transpose(1, 2).reshape(1, 1, -1)
        )
        spatial_x = self.spatial_fc_1(
            x + self.pos_embed_3.repeat(1, 32, 1).transpose(1, 2).reshape(1, 1, -1)
        )

        # reshape into 32×32 maps
        support_feat_out = spatial_support_feature.view(1, -1, 32, 32)
        support_summary_out = spatial_support_summary.view(1, -1, 32, 32)
        x_out = spatial_x.view(1, -1, 32, 32)

        recon_support = support_feat_out.clone()
        recon_x = x_out.clone()

        # -----------------------------
        # SPATIAL MULTI-SCALE (fixed)
        # -----------------------------
        merge_scales_space = []
        H = W = 32

        for i in range(self.spatial_scales):
            scale_h = max(1, H // (2 ** i))
            scale_w = max(1, W // (2 ** i))

            pool = lambda t: nn.AdaptiveAvgPool2d((scale_h, scale_w)).to(x.device)(t)

            sfo = pool(support_feat_out)
            sso = pool(support_summary_out)
            xot = pool(x_out)

            merged = torch.cat([sfo, sso, xot], dim=1)

            compress = nn.Sequential(
                nn.Conv2d(merged.shape[1], x_out.shape[1], kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ).to(x.device)

            fused = compress(merged)
            fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)

            merge_scales_space.append(fused)

        merge_scales_space = torch.stack(merge_scales_space, dim=0)
        spatial_w = torch.softmax(self.spatial_scale_weights, dim=0).view(-1, 1, 1, 1, 1)
        spatial_out = (spatial_w * merge_scales_space).sum(dim=0)
        spatial_out = spatial_out.view(1, -1, 1024)

        # -----------------------------
        # TEMPORAL MULTI-SCALE (fixed)
        # -----------------------------
        support_feat_out = self.attention(support_feature)[0] + support_feature
        support_summary_out = self.attention(support_summary)[0] + support_summary
        support_strengthen = torch.bmm(support_feature, support_summary.transpose(1, 2))

        up_fc = nn.Linear(support_strengthen.shape[-1], self.num_feature).to(x.device)
        support_updim = up_fc(support_strengthen)

        x_out2 = self.attention(x)[0] + x

        merge_scales_tpl = []

        row_sfo = support_feat_out.shape[1]
        row_sso = support_summary_out.shape[1]
        row_sup = support_updim.shape[1]
        row_xot = x_out2.shape[1]
        col = support_feat_out.shape[2]

        for i in range(self.temporal_scales):
            pool = lambda r, t: nn.AdaptiveAvgPool2d((r, col)).to(x.device)(t)

            sfo = pool(row_sfo, support_feat_out).unsqueeze(0)
            sso = pool(row_sso, support_summary_out).unsqueeze(0)
            sup = pool(row_sup, support_updim).unsqueeze(0)
            xot = pool(row_xot, x_out2).unsqueeze(0)

            merged = torch.cat([sfo, sso, sup, xot], dim=2)
            merged = F.interpolate(merged, size=(x.shape[1], x.shape[2]), mode="bilinear", align_corners=False)
            merge_scales_tpl.append(merged)

            row_sfo = max(1, row_sfo // 2)
            row_sso = max(1, row_sso // 2)
            row_sup = max(1, row_sup // 2)
            row_xot = max(1, row_xot // 2)

        merge_scales_tpl = torch.stack(merge_scales_tpl, dim=0)
        temp_w = torch.softmax(self.temporal_scale_weights, dim=0).view(-1, 1, 1, 1)
        temporal_out = (temp_w * merge_scales_tpl).sum(dim=0)

        while temporal_out.dim() > 3:
            temporal_out = temporal_out.squeeze(0)
        if temporal_out.dim() == 2:
            temporal_out = temporal_out.unsqueeze(0)

        # -----------------------------
        # MERGE TEMPORAL + SPATIAL
        # -----------------------------
        merged_all = torch.cat([temporal_out, spatial_out], dim=2)
        fc_merge = nn.Linear(merged_all.shape[-1], self.num_feature).to(x.device)
        merged_all = fc_merge(merged_all)

        merged_all = self.merge_extractor(merged_all)
        out = self.fc1(merged_all)

        seq_len = x.shape[1]
        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)
        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)

        recon_x = self.reconstruction(recon_x.view(1, -1, self.num_feature))
        recon_support = self.reconstruction(recon_support.view(1, -1, self.num_feature))

        return pred_cls, pred_loc, pred_ctr, recon_x, recon_support

    def predict(self, seq, support_seq, support_summary):
        pred_cls, pred_loc, pred_ctr, _, _ = self(seq, support_seq, support_summary)
        pred_cls = pred_cls * pred_ctr
        pred_cls /= pred_cls.max() + 1e-8
        return pred_cls, vs_helper.offset2bbox(pred_loc)
