import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ['DSASFFV54', 'DSASFFHead', 'scale_consistency_loss', 'entropy_regularization']


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSCConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, act=True):
        super().__init__()
        self.depthwise = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1, dilation=d, bias=False)
        self.pointwise = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class RFBLite(nn.Module):
    """轻量化多分支空洞卷积感受野增强 (共享版本)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter = max(out_channels // 4, 1)
        self.branch0 = nn.Sequential(Conv(in_channels, inter, 1), Conv(inter, inter, 3, d=1))
        self.branch1 = nn.Sequential(Conv(in_channels, inter, 1), Conv(inter, inter, 3, d=3))
        self.branch2 = nn.Sequential(Conv(in_channels, inter, 1), Conv(inter, inter, 3, d=5))
        self.branch3 = nn.Sequential(Conv(in_channels, inter, 1), Conv(inter, inter, 3, d=7))
        self.fuse = Conv(inter * 4, out_channels, 1)
        self.shortcut = Conv(in_channels, out_channels, 1, act=False)

    def forward(self, x):
        out = torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        out = self.fuse(out)
        return F.silu(out + self.shortcut(x))


class ScaleCompetitionAttention(nn.Module):
    """尺度竞争注意力 + 可选通道-尺度交互"""
    def __init__(self, channels, use_channel_scale_interaction=False):
        super().__init__()
        mid_channels = max(channels // 8, 1)
        self.spatial_score = nn.Sequential(
            Conv(channels, mid_channels, 1),
            Conv(mid_channels, 1, 1, act=False)
        )
        self.use_interaction = use_channel_scale_interaction
        if use_channel_scale_interaction:
            # 轻量通道-尺度门控: [B, 3, 1, 1]
            self.channel_scale_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Conv(channels, mid_channels, 1),
                Conv(mid_channels, 3, 1, act=False),  # 输出 3 个尺度因子
                nn.Sigmoid()
            )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # spatial competition weights: [B,3,H,W]
        spatial_scores = torch.stack([self.spatial_score(f) for f in feats], dim=1)  # [B,3,1,H,W]
        spatial_weights = torch.softmax(spatial_scores, dim=1).squeeze(2)            # [B,3,H,W]

        if self.use_interaction:
            # channel-scale interaction: [B,3,1,1]
            # 使用第一个特征作为代表计算全局门控 (或可平均所有特征)
            scale_gates = self.channel_scale_gate(feats[0])  # [B,3,1,1]
            weights = spatial_weights * scale_gates          # 广播乘法
            # 重新归一化保持和为1 (可选)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            return weights
        return spatial_weights


class DSASFFV54(nn.Module):
    """
    DSASFF-V5.4: 高效动态融合模块
    - Shared RFBLite stem (大幅降低 FLOPs)
    - 每个尺度独立 level_embed (保留尺度身份)
    - 竞争注意力 (可选通道-尺度交互)
    - 后融合精炼
    """
    def __init__(self, hidden_ch: int = 256, use_channel_scale_interaction=False):
        super().__init__()
        self.hidden_ch = hidden_ch
        # 层级嵌入 (每个尺度独立)
        self.level_embed = nn.Parameter(torch.randn(3, hidden_ch, 1, 1))
        # 竞争注意力
        self.attn = ScaleCompetitionAttention(hidden_ch, use_channel_scale_interaction)
        # 精炼
        self.refine = nn.Sequential(
            Conv(hidden_ch, hidden_ch, 3),
            Conv(hidden_ch, hidden_ch, 3)
        )
        self.expand = DSCConv(hidden_ch, hidden_ch, 3, 1)

    @staticmethod
    def _resize_to(x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

    def forward(self, base_feats: List[torch.Tensor], target_level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        base_feats: 已经过 reduce+shared_stem 的特征列表 (各分辨率不同)
        target_level: 0,1,2 目标输出层级 (仅用于确定目标分辨率，不再用于嵌入)
        """
        target_size = base_feats[target_level].shape[-2:]

        # 1. 对齐所有特征到目标分辨率
        aligned = [self._resize_to(f, target_size) for f in base_feats]

        # 2. 添加层级嵌入 (每个尺度加自己的嵌入，保留尺度身份)
        for i in range(3):
            aligned[i] = aligned[i] + self.level_embed[i]

        # 3. 竞争注意力
        weights = self.attn(aligned)                # [B,3,H,W]

        # 4. 动态融合
        fused = (aligned[0] * weights[:, 0:1] +
                 aligned[1] * weights[:, 1:2] +
                 aligned[2] * weights[:, 2:3])

        # 5. 精炼 + 扩展
        fused = self.refine(fused)
        fused = self.expand(fused)

        return fused, weights


class DSASFFHead(nn.Module):
    """
    DSASFF-V5.4 检测头：共享 stem，预计算基础特征
    """
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=1, ch=None, hidden_ch=256, use_channel_scale_interaction=False, *args, **kwargs):
        super().__init__()
        if isinstance(nc, (list, tuple)):
            nc = nc[0] if len(nc) else 1
        self.nc = int(nc)
        self.hidden_ch = hidden_ch
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.nl = 3

        self.proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer("stride", torch.tensor([8., 16., 32.]), persistent=False)

        if ch is None:
            ch = [256, 512, 1024]
        self.ch = ch

        # ----- 降维 (每个尺度独立，因为输入通道不同) -----
        self.reduce = nn.ModuleList([Conv(c, 64, 1) for c in ch])

        # ----- 共享 stem (所有尺度共用同一个 RFBLite) -----
        self.stem = RFBLite(64, hidden_ch)

        # ----- 融合模块 -----
        self.fusion = DSASFFV54(hidden_ch, use_channel_scale_interaction)

        # ----- 检测头卷积 -----
        c2 = max(16, hidden_ch // 4, self.reg_max * 4)
        c3 = max(hidden_ch, min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(hidden_ch, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for _ in range(self.nl)
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(hidden_ch, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, self.nc, 1)
            ) for _ in range(self.nl)
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self._fused_feats = None
        self._weights_list = None

    def forward(self, x: List[torch.Tensor], return_weights=False):
        if len(x) != 3:
            raise ValueError(f"DSASFFHead expects 3 feature maps, got {len(x)}")

        # 清除上一次 forward 的缓存，避免验证阶段误用训练阶段的特征。
        self._fused_feats = None
        self._weights_list = None

        # ---------- 预计算基础特征 (reduce + shared stem) ----------
        base_feats = []
        for i, feat in enumerate(x):
            reduced = self.reduce[i](feat)          # [B,32,H_i,W_i]
            stemmed = self.stem(reduced)            # [B,hidden_ch,H_i,W_i] (共享 stem)
            base_feats.append(stemmed)

        # ---------- 对每个目标层级进行融合 ----------
        fused_feats = []
        weights_list = []
        for level in range(self.nl):
            fused, w = self.fusion(base_feats, target_level=level)
            fused_feats.append(fused)
            weights_list.append(w)

        # BaseModel.loss() 会读取这两个缓存计算辅助损失。
        if self.training:
            self._fused_feats = fused_feats
            self._weights_list = weights_list

        # ---------- 检测头预测 ----------
        outputs = []
        for i in range(self.nl):
            pred = torch.cat([self.cv2[i](fused_feats[i]), self.cv3[i](fused_feats[i])], dim=1)
            outputs.append(pred)

        if self.training:
            if return_weights:
                return outputs, weights_list
            return outputs

        # 推理模式
        shape = fused_feats[0].shape
        if self.dynamic or self.shape != shape or self.anchors.numel() == 0:
            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(outputs, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in outputs], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, outputs)

    def bias_init(self):
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:self.nc] = math.log(
                5.0 / self.nc / (640.0 / float(s)) ** 2
            )


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


# ========== 对称尺度一致性损失 ==========
def scale_consistency_loss(fused_feats: List[torch.Tensor], use_cosine=True) -> torch.Tensor:
    """
    对称 L1 损失 + 可选的对称余弦损失
    """
    spatial_loss = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            size_i = fused_feats[i].shape[-2:]
            size_j = fused_feats[j].shape[-2:]
            # 双向对齐损失 (对称)
            if size_i != size_j:
                # i -> j
                f_i_to_j = F.interpolate(fused_feats[i], size=size_j, mode='bilinear', align_corners=False)
                spatial_loss += F.l1_loss(f_i_to_j, fused_feats[j])
                # j -> i
                f_j_to_i = F.interpolate(fused_feats[j], size=size_i, mode='bilinear', align_corners=False)
                spatial_loss += F.l1_loss(f_j_to_i, fused_feats[i])
            else:
                spatial_loss += F.l1_loss(fused_feats[i], fused_feats[j])
    spatial_loss = spatial_loss / 6.0  # 共有6个方向 (每对两个方向)

    if not use_cosine:
        return spatial_loss

    # 对称余弦损失
    pooled = [F.adaptive_avg_pool2d(f, 1).view(f.size(0), -1) for f in fused_feats]
    cosine_loss = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            cos_sim_ij = F.cosine_similarity(pooled[i], pooled[j], dim=-1)
            cos_sim_ji = F.cosine_similarity(pooled[j], pooled[i], dim=-1)  # 对称
            cosine_loss += (1 - cos_sim_ij).mean() + (1 - cos_sim_ji).mean()
    cosine_loss = cosine_loss / 6.0
    return spatial_loss + 0.5 * cosine_loss


def entropy_regularization(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    熵正则化，鼓励注意力权重分散，防止 collapse 到单一尺度。
    weights: [B, 3, H, W] 尺度竞争权重 (已归一化)
    """
    # 避免 log(0)
    weights = weights.clamp_min(eps)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(weights * weights.log()).sum(dim=1)
    min_entropy = 0.5 * math.log(max(weights.shape[1], 2))
    return F.relu(min_entropy - entropy).mean()
