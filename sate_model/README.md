# Satellite2PV 模型参数说明文档

## 模型概述

Satellite2PV 是一个基于BEVFormer架构优化的逆投影变换模型，专门用于将卫星地图的BEV（鸟瞰图）特征转换为目标视角的PV（透视视图）图像。该模型包含两个主要组件：

1. **SatelliteEncoder** - 卫星地图特征编码器
2. **InverseProjectionTransformer** - 逆投影变换器

---

## 核心参数详解

### 1. Satellite2PV 主模型参数

```python
class Satellite2PV(BaseModule):
    def __init__(self,
                 satellite_encoder_cfg,    # 卫星编码器配置
                 inverse_projection_cfg,   # 逆投影变换器配置
                 bev_size=(16, 16),      # BEV特征图尺寸 (H, W)
                 pv_size=(68, 120),     # PV特征尺寸 (H, W)
                 init_cfg=None):
```

**参数说明：**
- `satellite_encoder_cfg`: 卫星编码器配置字典
- `inverse_projection_cfg`: 逆投影变换器配置字典  
- `bev_size`: BEV特征图尺寸，
- `pv_size`: PVPV特征尺寸

---

### 2. SatelliteEncoder 卫星编码器参数

```python
class SatelliteEncoder(BaseModule):
    def __init__(self,
                 backbone_cfg=None,    # 主干网络配置
                 neck_cfg=None,        # 颈部网络配置
                 init_cfg=None):
```

#### 2.1 默认主干网络配置 (ResNet50)
```python
backbone_cfg = dict(
    type='ResNet',
    depth=50,                    # ResNet深度 (50, 101, 152)
    num_stages=4,                # 阶段数
    out_indices=(3,),            # 输出阶段索引 (0,1,2,3)
    frozen_stages=1,             # 冻结阶段数 (加速训练)
    norm_cfg=dict(type='BN2d', requires_grad=False),  # 批归一化配置
    norm_eval=True,              # 评估时冻结BN统计量
    style='caffe',               # 网络风格 (caffe/pytorch)
    with_cp=True,                # 使用checkpoint节省显存
    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),  # 可变形卷积
    stage_with_dcn=(False, False, True, True)  # 哪些阶段使用DCN
)
```

#### 2.2 默认颈部网络配置 (FPN)
```python
neck_cfg = dict(
    type='FPN',
    in_channels=[2048],          # 输入通道数 (ResNet50最后一层)
    out_channels=256,            # 输出通道数
    start_level=0,               # 起始特征层级
    add_extra_convs='on_output', # 额外卷积位置
    num_outs=1,                  # 输出特征图数量
    relu_before_extra_convs=True # 额外卷积前使用ReLU
)
```

---

### 3. InverseProjectionTransformer 逆投影变换器参数

```python
class InverseProjectionTransformer(BaseModule):
    def __init__(self,
                 hidden_channels=256,      # 隐藏层通道数
                 num_heads=8,              # 注意力头数
                 num_layers=2,             # 变换器层数
                 dropout=0.1,              # Dropout率
                 pv_size=[68,120,100],     # PV尺寸 [H, W, D] (高度, 宽度, 深度)
                 bev_range=(0,-50,100,50), # BEV范围 (x_start, y_start, x_end, y_end)
                 positional_encoding=dict( # 位置编码配置
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 norm_cfg=dict(type='LN'), # 归一化配置
                 init_cfg=None):
```

**关键参数说明：**

#### 3.1 核心维度参数
- `hidden_channels=256`: 特征维度，与FPN输出通道保持一致
- `num_heads=8`: 多头注意力头数，影响模型表达能力
- `num_layers=2`: 变换器层数，控制模型深度
- `pv_size=[68,120,100]`: PV空间尺寸 [高度, 宽度, 深度采样点数]

#### 3.2 空间范围参数
- `bev_range=(0,-50,100,50)`: BEV空间范围 (x_start, y_start, x_end, y_end)
  - 对应世界坐标系范围：x∈[0,100], y∈[-50,50] (单位：米)
  - 用于将世界坐标归一化到BEV网格

#### 3.3 位置编码参数
```python
positional_encoding=dict(
    type='SinePositionalEncoding',  # 正弦位置编码
    num_feats=128,                  # 位置编码特征维度
    normalize=True                  # 是否归一化
)
```

#### 3.4 输出投影配置 (重要修复)
```python
# 修复后的输出投影 - 使用BatchNorm2d处理4D张量
self.output_projection = nn.Sequential(
    build_conv_layer(dict(type='Conv2d'), hidden_channels, hidden_channels, 
                     kernel_size=3, stride=1, padding=1, bias=False),
    build_norm_layer(dict(type='BN2d'), hidden_channels)[1],  # BatchNorm2d
    nn.ReLU(inplace=True)
)
```
**修复说明：** 将原来的LayerNorm改为BatchNorm2d，因为：
- LayerNorm期望输入形状为 `[*, 256]` (2D张量)
- 实际输入形状为 `[2, 256, 68, 120]` (4D张量)
- BatchNorm2d专门设计用于处理4D特征图

---

### 4. InverseProjectionLayer 逆投影层参数

```python
class InverseProjectionLayer(BaseModule):
    def __init__(self,
                 hidden_channels,          # 隐藏层通道数
                 num_heads,                # 注意力头数
                 dropout=0.1,              # Dropout率
                 num_pillar_in_depth=100,  # 深度维度采样点数
                 norm_cfg=dict(type='LN'), # 归一化配置
                 init_cfg=None):
```

**组件说明：**

#### 4.1 自注意力模块
```python
self.self_attn = DeformableSelfAttentionInverse(
    embed_dims=hidden_channels,
    num_heads=num_heads,
    num_levels=1,        # 单尺度特征
    num_points=8,        # 每个注意力头的采样点数
    dropout=dropout,
    batch_first=True     # 批次维度在前
)
```

**关键特性：**
- **单帧处理**: 只做自注意力过程，没有多帧融合
- **可变形注意力**: 自适应采样关键位置
- **极坐标初始化**: 采样偏移量使用极坐标分布初始化

#### 4.2 空间交叉注意力
```python
self.spatial_cross_attn = SpatialCrossAttentionInverse(
    embed_dims=hidden_channels,
    num_heads=num_heads,
    dropout=dropout
)
```

#### 4.3 前馈网络
```python
self.ffn = nn.Sequential(
    nn.Linear(hidden_channels, hidden_channels * 4),  # 扩展4倍
    nn.ReLU(inplace=True),
    nn.Dropout(dropout),
    nn.Linear(hidden_channels * 4, hidden_channels),  # 压缩回原维度
    nn.Dropout(dropout)
)
```

---

### 5. 注意力模块参数

#### 5.1 DeformableSelfAttentionInverse (自注意力)
```python
class DeformableSelfAttentionInverse(BaseModule):
    def __init__(self,
                 embed_dims=256,    # 嵌入维度
                 num_heads=8,       # 注意力头数
                 num_levels=1,      # 特征层级数
                 num_points=8,      # 每个头的采样点数
                 dropout=0.1,       # Dropout率
                 batch_first=True,  # 批次维度在前
                 norm_cfg=None,
                 init_cfg=None):
```

#### 5.2 MSDeformableAttention3DInverse (3D可变形注意力)
```python
class MSDeformableAttention3DInverse(BaseModule):
    def __init__(self,
                 embed_dims=256,      # 嵌入维度
                 num_heads=8,         # 注意力头数
                 num_levels=1,        # 特征层级数
                 num_points=500,      # 采样点数 (深度维度)
                 im2col_step=64,      # im2col步长 (CUDA优化)
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
```

---

## 默认配置示例

### 轻量级配置 (默认)
```python
default_config = {
    'satellite_encoder': {
        'backbone_cfg': { ... },  # ResNet50配置
        'neck_cfg': { ... }       # FPN配置
    },
    'inverse_projection': {
        'hidden_channels': 256,
        'num_heads': 8,
        'num_layers': 2
    },
    'bev_size': (16, 16),    # 小尺寸BEV
    'pv_size': (68, 120)     # 小尺寸PV
}
```

---
