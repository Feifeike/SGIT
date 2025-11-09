import torch
import torch.nn as nn
from mmengine.registry import MODELS

@MODELS.register_module()
class SinePositionalEncoding(nn.Module):
    """参考官方库实现，支持 1D/2D 输入，自动适配维度"""
    def __init__(self, num_feats, normalize=False, temperature=10000.0, scale=2 * torch.pi):
        super().__init__()
        self.num_feats = num_feats  # 编码维度的一半（输出为 2*num_feats）
        self.normalize = normalize  # 是否归一化位置坐标到 [0,1]
        self.temperature = temperature
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: 输入特征，支持 3D (B, L, C) 或 4D (B, C, H, W) 或 2D (B, H, W)
        Returns:
            pos_encoding: 与输入空间/序列维度匹配的位置编码
        """
        # 动态解析输入维度（忽略 batch 维度）
        if x.dim() == 4:  # (B, C, H, W) -> 2D 特征图
            _, _, H, W = x.shape
            mask = torch.ones((H, W), device=x.device)  # 假设无掩码，全为有效区域
        elif x.dim() == 3:  # 可能是 (B, L, C) 1D 序列 或 (B, H, W) 2D 无通道
            if x.size(1) > x.size(2):  # 简单判断：长序列视为 1D
                B, L, _ = x.shape
                mask = torch.ones((B, L), device=x.device)  # (B, L)
                return self._1d_encoding(mask)
            else:  # 视为 2D (B, H, W)
                B, H, W = x.shape
                mask = torch.ones((H, W), device=x.device)
        elif x.dim() == 2:  # (B, L) 1D 掩码
            mask = x  # 直接使用输入作为掩码（常用于 1D 序列）
            return self._1d_encoding(mask)
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")

        # 2D 编码逻辑
        return self._2d_encoding(mask)

    def _2d_encoding(self, mask):
        """处理 2D 特征图的位置编码 (H, W)"""
        H, W = mask.shape
        device = mask.device

        # 生成 (H, W) 的坐标网格
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)  # (H, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)  # (H, W)

        if self.normalize:
            y_embed = y_embed / (H - 1 + 1e-6) * self.scale
            x_embed = x_embed / (W - 1 + 1e-6) * self.scale

        # 计算正弦/余弦编码
        dim_t = torch.arange(self.num_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature **(2 * (dim_t // 2) / self.num_feats)  # (D/2,)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # (H, W, D/2)
        pos_y = y_embed.unsqueeze(-1) / dim_t  # (H, W, D/2)

        # 交替拼接正弦和余弦
        pos_x = torch.stack([pos_x[..., ::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(2)  # (H, W, D)
        pos_y = torch.stack([pos_y[..., ::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(2)  # (H, W, D)

        # 拼接 x 和 y 方向编码，扩展 batch 维度
        pos_encoding = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1).unsqueeze(0)  # (1, 2D, H, W)
        return pos_encoding

    def _1d_encoding(self, mask):
        """处理 1D 序列的位置编码 (B, L)"""
        B, L = mask.shape
        device = mask.device

        # 生成 (L,) 的位置坐标
        x_embed = torch.arange(L, device=device).float()  # (L,)

        if self.normalize:
            x_embed = x_embed / (L - 1 + 1e-6) * self.scale

        dim_t = torch.arange(self.num_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature** (2 * (dim_t // 2) / self.num_feats)  # (D/2,)

        pos_x = x_embed.unsqueeze(-1) / dim_t  # (L, D/2)
        pos_x = torch.stack([pos_x[..., ::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(1)  # (L, D)

        # 扩展 batch 维度
        pos_encoding = pos_x.unsqueeze(0).permute(0, 2, 1).repeat(B, 1, 1)  # (B, D, L)
        return pos_encoding