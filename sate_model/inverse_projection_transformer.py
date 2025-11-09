import torch
import torch.nn as nn
import math
import warnings
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, xavier_init, constant_init
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch


class InverseProjectionTransformer(BaseModule):
    """逆投影变换器 - 基于BEVFormer rough head pipeline优化
    
    将BEV特征投影到目标PV视角，实现从卫星地图到透视视图的转换
    参考BEVFormerRoughnessHead的完整pipeline设计
    """
    
    def __init__(self,
                 hidden_channels=256,
                 num_heads=8,
                 num_layers=2,
                 dropout=0.1,
                 pv_size=[68,120,100],  # H, W, D
                 bev_range=(0,-50,100,50),
                 positional_encoding=dict(
                     type='custom_pos_encoding.SinePositionalEncoding',
                     num_feats=160,
                     normalize=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(InverseProjectionTransformer, self).__init__(init_cfg)

        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pv_size = pv_size
        self.pv_h = pv_size[0]  
        self.pv_w = pv_size[1]
        self.num_pillar_in_depth = pv_size[2]
        self.bev_range = bev_range

        # PV查询嵌入 (参考BEVFormerRoughnessHead.bev_embedding)
        self.pv_embedding = nn.Embedding(pv_size[0] * pv_size[1], self.hidden_channels)

        # 位置编码 (参考BEVFormerRoughnessHead.positional_encoding)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        
        # 变换器层
        self.transformer_layers = nn.ModuleList([
            InverseProjectionLayer(
                hidden_channels,
                num_heads,
                dropout=dropout,
                num_pillar_in_depth=self.num_pillar_in_depth,
                norm_cfg=norm_cfg)
            for _ in range(num_layers)
        ])
        
        # 输出投影 - 使用BatchNorm2d而不是LayerNorm来处理4D张量
        self.output_projection = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            build_norm_layer(dict(type='BN2d'), hidden_channels)[1],
            nn.ReLU(inplace=True)
        )
    
    def forward(self, bev_features, camera_params):
        """前向传播 - 参考BEVFormerRoughnessHead.forward流程
        
        Args:
            bev_features (Tensor): BEV特征 [B, C, H_bev, W_bev]
            camera_params (dict): 相机参数，包含变换矩阵
            bev_range : BEV大小
            
        Returns:
            Tensor: PV特征 [B, C, H_pv, W_pv]
        """
        batch_size, C, H_bev, W_bev = bev_features.shape
        dtype = bev_features.dtype
        device = bev_features.device
        
        # 1. 准备PV查询 (参考BEVFormerRoughnessHead的BEV查询构建)
        pv_queries = self.pv_embedding.weight.to(dtype)  # [pv_h * pv_w, hidden_channels]
        
        # 2. 准备PV位置编码 (参考BEVFormerRoughnessHead的位置编码)
        pv_mask = torch.zeros((batch_size, self.pv_h, self.pv_w), device=device).to(dtype)
        pv_pos = self.positional_encoding(pv_mask).to(dtype)  # [B, hidden_channels, pv_h, pv_w]
        
        # 4. 变换器处理 (参考PerceptionTransformer_rough.get_bev_features)
        # 重塑查询和位置为序列格式
        pv_queries_seq = pv_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, pv_h * pv_w, hidden_channels]
        pv_pos_seq = pv_pos.flatten(2).transpose(1, 2)  # 更可靠的展平方式

        # 重塑BEV特征为序列格式 (参考BEVFormerEncoder_rough的特征展平)
        bev_seq = bev_features.reshape(batch_size, self.hidden_channels, -1).transpose(1, 2)  # [B, N_bev, hidden_channels]
        
        # 通过变换器层 (参考BEVFormerEncoder_rough.forward)
        for layer in self.transformer_layers:
            pv_queries_seq = layer(
                pv_queries_seq, 
                bev_seq, 
                pv_pos_seq,
                camera_params,
                bev_range=self.bev_range,
                pv_size=self.pv_size,
                bev_size=[H_bev,W_bev],
            )
        
        # 5. 重塑回空间格式
        pv_features = pv_queries_seq.transpose(1, 2).view(batch_size, self.hidden_channels, self.pv_h, self.pv_w)
        
        # 6. 输出投影
        pv_features = self.output_projection(pv_features)
        
        return pv_features


class InverseProjectionLayer(BaseModule):
    """逆投影变换层,参考BEVFormer的rough_encoder实现"""
    
    def __init__(self,
                 hidden_channels,
                 num_heads,
                 dropout=0.1,
                 num_pillar_in_depth=100,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(InverseProjectionLayer, self).__init__(init_cfg)
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_pillar_in_depth = num_pillar_in_depth
        
        # 自注意力（使用专门的自注意力deformable类）
        self.self_attn = DeformableSelfAttentionInverse(
            embed_dims=hidden_channels,
            num_heads=num_heads,
            num_levels=1,
            num_points=8,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = build_norm_layer(norm_cfg, hidden_channels)[1]
        
        # 空间交叉注意力（BEV到PV）
        self.spatial_cross_attn = SpatialCrossAttentionInverse(
            embed_dims=hidden_channels,
            num_heads=num_heads,
            dropout=dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.Dropout(dropout)
        )
        self.ffn_norm = build_norm_layer(norm_cfg, hidden_channels)[1]
    
    @staticmethod
    def get_reference_points(pv_size, num_pillar_in_depth=100, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """获取3D参考点，用于空间交叉注意力
        
        Args:
            H, W: PV空间的空间形状
            D: 深度维度 m
            num_pillar_in_depth: 每个depth中的采样点数
            dim: 维度类型 ('3d' 或 '2d')
            bs: batch size
            device: 设备
            dtype: 数据类型
            
        Returns:
            Tensor: 3D参考点
        """
        H, W, D = pv_size
        if dim == '3d':
            # 在PV空间生成3D参考点
            zs = torch.linspace(0, D, num_pillar_in_depth, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_pillar_in_depth, H, W) / D
            xs = torch.linspace(0, W, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_pillar_in_depth, H, W) / W
            ys = torch.linspace(0, H, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_pillar_in_depth, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)  # [num_points, H, W, 3]
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)  # [num_points, H*W, 3]
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  # [bs, num_points, H*W, 3]
            return ref_3d
        
        elif dim == '2d':
            # 2D参考点用于自注意力
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0, H, H, dtype=dtype, device=device),
                torch.linspace(0, W, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
    
    def point_sampling(self, reference_points, pv_size, camera_params, bev_range):
        """点采样 - 反向投影过程
        
        将图像上的点先还原到相机坐标系下,根据world2pv变换矩阵投影到世界坐标系下,
        根据bev_size将世界坐标转到bev坐标上
        
        Args:
            reference_points: 3D参考点 [B, n, N_pv, 3] (图像坐标系，归一化坐标)
            pv_size: PV尺寸 (H, W, D)
            camera_params: 相机参数，包含变换矩阵
            bev_range: BEV范围 (x_start, y_start, x_end, y_end),单位mm
            
        Returns:
            tuple: (投影后的BEV参考点, 有效掩码)
        """
        # 获取相机变换矩阵 (PV到BEV的逆变换)
        world2pv = camera_params.get('world2pv', None)
        world2pv = reference_points.new_tensor(world2pv)  # [B, 4, 4]
        pv2bev = torch.inverse(world2pv)

        reference_points = reference_points.clone()
        
        batch_size, n, num_pv, _ = reference_points.shape
        H, W, D = pv_size
        
        # 1. 将归一化图像坐标映射到像素坐标
        # 图像坐标系: (u, v, d) 其中u是宽度方向，v是高度方向
        reference_points[..., 0:1] = reference_points[..., 0:1] * W  # u: [0, W]
        reference_points[..., 1:2] = reference_points[..., 1:2] * H  # v: [0, H] 
        reference_points[..., 2:3] = reference_points[..., 2:3] * D * 1000 # d: [0, D] (深度) m ->mm

        # 2. 将图像坐标(u, v, d)还原到图像坐标系(x_cam, y_cam, z_cam)
        x_cam = reference_points[..., 0:1] * reference_points[..., 2:3]
        y_cam = reference_points[..., 1:2] * reference_points[..., 2:3]
        z_cam = reference_points[..., 2:3]  # 深度作为相机坐标系的z坐标
        
        # 3. 构建相机坐标系下的3D点 [x_cam, y_cam, z_cam, 1]
        # 这里我们假设相机坐标系原点在图像中心，所以需要将像素坐标转换到相机坐标系
        # 由于pv2bev矩阵已经包含了相机内参，我们直接使用(u, v, d)构建齐次坐标
        camera_coords = torch.cat([
            x_cam,  # x_cam
            y_cam,  # y_cam  
            z_cam,  # z_cam
            torch.ones_like(reference_points[..., :1])  # 齐次坐标
        ], dim=-1)  # [B, n, N_pv, 4]
        
        # 4. 应用pv2bev变换矩阵将相机坐标系转换到世界坐标系
        # pv2bev: [B, 4, 4] 相机坐标系到世界坐标系的变换矩阵
        camera_coords = camera_coords.view(batch_size, n * num_pv, 4).transpose(1, 2)  # [B, 4, n*N_pv]
        world_coords = torch.matmul(pv2bev, camera_coords)  # [B, 4, n*N_pv]
        world_coords = world_coords.transpose(1, 2).view(batch_size, n, num_pv, 4)  # [B, n, N_pv, 4]
        
        # 6. 将世界坐标转换到BEV坐标
        # 假设世界坐标范围已知，这里需要根据实际的点云范围进行归一化
        # 由于bev图原点在图的中下边，我们需要特殊处理y坐标
        x_start, y_start, x_end, y_end = bev_range
        
        # 计算世界坐标在BEV空间中的归一化坐标(中心点在bev中下边界)
        bev_h = (x_end - world_coords[..., 0]) / (x_end - x_start)  # x: [0, 1]
        bev_w = (y_end - world_coords[..., 1]) / (y_end - y_start)  # y: [0, 1]
        
        # 组合BEV坐标
        reference_points_bev = torch.stack([bev_h, bev_w], dim=-1)  # [B, n, N_pv, 2]
        
        # 7. 生成有效掩码 - 过滤越界点
        pv_mask = (
            (reference_points_bev[..., 0] >= 0.0) & 
            (reference_points_bev[..., 0] <= 1.0) &
            (reference_points_bev[..., 1] >= 0.0) & 
            (reference_points_bev[..., 1] <= 1.0) 
        )  # [B, n, N_pv]
        
        # 重塑掩码形状以匹配参考点
        pv_mask = pv_mask.permute(0,2,1)  # [B, n, N_pv, 1]
        
        return reference_points_bev, pv_mask
    
    def forward(self, pv_query_seq, bev_seq, query_pos, camera_params, bev_range, pv_size, bev_size):
        """前向传播
        
        Args:
            pv_query_seq (Tensor): PV查询序列 [B, N_pv, C]
            bev_seq (Tensor): BEV特征序列 [B, N_bev, C]
            camera_params (dict): 相机参数
            bev_range:  BEV范围 (x_start, y_start, x_end, y_end),单位mm
            pv_size: PV (W,H,D)
            
        Returns:
            Tensor: 更新后的PV查询序列 [B, N_pv, C]
        """
        batch_size, num_pv, channels = pv_query_seq.shape
        bev_h, bev_w = bev_size
        
        # 1. 生成3D PV参考点
        ref_3d = self.get_reference_points(
            pv_size=pv_size,
            num_pillar_in_depth=self.num_pillar_in_depth,
            dim='3d', 
            bs=batch_size, 
            device=pv_query_seq.device, 
            dtype=pv_query_seq.dtype)
        
        # 生成2D参考点用于自注意力
        ref_2d = self.get_reference_points(
            pv_size,
            dim='2d',
            bs=batch_size,
            device=pv_query_seq.device,
            dtype=pv_query_seq.dtype
        )
        
        # 2. 使用相机变换矩阵将3D PV点投影到BEV视角
        reference_points_pv, pv_mask = self.point_sampling(
            ref_3d, pv_size, camera_params, bev_range)

        # 3. 自注意力（使用可变形注意力）
        self_attn_out = self.self_attn(
            query=pv_query_seq,
            key=pv_query_seq,
            value=pv_query_seq,
            query_pos=query_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[pv_size[0], pv_size[1]]], device=pv_query_seq.device),
            level_start_index=torch.tensor([0], device=pv_query_seq.device)
        )
        pv_query_seq = self.self_attn_norm(pv_query_seq + self_attn_out)
        
        # 4. 空间交叉注意力（使用几何投影）     
        cross_attn_out = self.spatial_cross_attn(
            query=pv_query_seq,
            key=bev_seq,
            value=bev_seq,
            reference_points_pv=reference_points_pv,
            pv_mask=pv_mask,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=pv_query_seq.device),
            level_start_index=torch.tensor([0], device=pv_query_seq.device)
        )
        pv_query_seq = self.self_attn_norm(pv_query_seq + cross_attn_out)
        
        # 5. 前馈网络
        ffn_out = self.ffn(pv_query_seq)
        pv_query_seq = self.ffn_norm(pv_query_seq + ffn_out)
        
        return pv_query_seq


class DeformableSelfAttentionInverse(BaseModule):
    """逆投影自注意力模块 - 专门处理自注意力的deformable类
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=8,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super(DeformableSelfAttentionInverse, self).__init__(init_cfg)
        
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        
        # 采样偏移量预测 - 简化为单帧
        self.sampling_offsets = nn.Linear(
            embed_dims, 
            num_heads * num_levels * num_points * 2)
        
        # 注意力权重预测
        self.attention_weights = nn.Linear(
            embed_dims,
            num_heads * num_levels * num_points)
        
        # 值投影
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        constant_init(self.sampling_offsets, 0.)
        
        # 初始化采样偏移量 - 使用极坐标初始化
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / 
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        """前向传播 - 自注意力机制
        
        Args:
            query (Tensor): 查询 [B, N_pv, C]
            key (Tensor): 键 [B, N_pv, C]
            value (Tensor): 值 [B, N_pv, C]
            identity (Tensor): 残差连接 [B, N_pv, C]
            query_pos (Tensor): 查询位置编码
            key_padding_mask (Tensor): 键填充掩码
            reference_points (Tensor): 参考点 [B, N_pv, num_levels, 2]
            spatial_shapes (Tensor): 空间形状 [num_levels, 2]
            level_start_index (Tensor): 层级起始索引 [num_levels]
            
        Returns:
            Tensor: 注意力输出 [B, N_pv, C]
        """
        # 处理初始情况
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        batch_size, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        
        # 值投影
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        
        # 重塑value为多头形式
        value = value.view(batch_size, num_value, self.num_heads, -1)
        
        # 预测采样偏移量
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # 预测注意力权重
        attention_weights = self.attention_weights(query).view(
            batch_size, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(batch_size, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        # 计算采样位置
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2, but got {reference_points.shape[-1]} instead.')
        
        # 应用可变形注意力
        if torch.cuda.is_available() and value.is_cuda:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        
        # 输出投影
        output = self.output_proj(output)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        return self.dropout(output) + identity


class SpatialCrossAttentionInverse(BaseModule):
    """逆投影空间交叉注意力模块
    
    基于BEVFormer的SpatialCrossAttention_rough，专门用于从BEV到PV的逆投影变换
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 bev_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(SpatialCrossAttentionInverse, self).__init__(init_cfg)
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.bev_range = bev_range
        self.batch_first = batch_first
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        # 可变形注意力模块
        self.deformable_attention = MSDeformableAttention3DInverse(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=1,  # 单尺度BEV特征
            num_points=500,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.init_weight()
    
    def init_weight(self):
        """权重初始化"""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    def forward(self,
                query,
                key,
                value,
                reference_points_pv,
                pv_mask,
                spatial_shapes,
                level_start_index,
                **kwargs):
        """前向传播 - PV采样过程
        
        Args:
            query (Tensor): PV查询 [B, N_pv, C]
            key (Tensor): BEV键 [B, N_bev, C]
            value (Tensor): BEV值 [B, N_bev, C]
            reference_points_pv (Tensor): 3D PV参考点 [B, D, N_pv, 3]
            pv_mask (Tensor): PV掩码 [B, D, N_pv]
            spatial_shapes (Tensor): 空间形状 [num_levels, 2]
            level_start_index (Tensor): 层级起始索引 [num_levels]
            
        Returns:
            Tensor: 注意力输出 [B, N_pv, C]
        """
        batch_size, num_pv, _ = query.shape
        _, num_bev, _ = key.shape
        
        # 保存输入残差
        inp_residual = query
        
        # 处理参考点形状
        D = reference_points_pv.size(1)  # 深度维度
        num_Z_anchors = D
        
        # 对pv_query进行rebatch处理
        # 获取有效查询的索引
        indexes = []
        for j in range(batch_size):
            index_query = pv_mask[j].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query)
        max_len = max([len(each) for each in indexes])
            
        # 创建rebatch的查询和参考点
        queries_rebatch = query.new_zeros([batch_size, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_pv.new_zeros([batch_size, max_len, num_Z_anchors, 2])
        
        # 填充rebatch张量
        for j in range(batch_size):
            queries_rebatch[j] = query[j, indexes[j]]
            reference_points_rebatch[j] = reference_points_pv[j, :, indexes[j]].permute(1, 0, 2)
        
        # 应用可变形注意力
        attn_output = self.deformable_attention(
            query=queries_rebatch,
            key=key,
            value=value,
            identity=inp_residual,
            reference_points=reference_points_rebatch,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )
        
        # 将结果重新映射回原始查询位置
        slots = torch.zeros_like(query)
        for j in range(batch_size):
            slots[j, indexes[j]] = attn_output[j]
        
        # 输出投影
        output = self.output_proj(slots)
        
        return self.dropout(output) + inp_residual


class MSDeformableAttention3DInverse(BaseModule):
    """逆投影3D可变形注意力模块 - 基于MSDeformableAttention3D_rough优化
    
    参考BEVFormer的MSDeformableAttention3D_rough实现，专门用于逆投影变换
    包含完整的输出投影和优化的CUDA实现
    """
    
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=500,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super(MSDeformableAttention3DInverse, self).__init__(init_cfg)
        
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.fp16_enabled = False
        
        # 检查dim_per_head是否为2的幂次方（CUDA优化）
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0
        
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')
        
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        
        # 采样偏移量预测
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        
        # 注意力权重预测
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        
        # 值投影
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        
        self.init_weights()
    
    def init_weights(self):
        """权重初始化"""
        constant_init(self.sampling_offsets, 0.)
        
        # 初始化采样偏移量 - 使用极坐标初始化
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / 
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                bev_mask=None,
                **kwargs):
        """前向传播
        
        Args:
            query (Tensor): 查询 [B, N_pv, C]
            key (Tensor): 键 [B, N_bev, C]
            value (Tensor): 值 [B, N_bev, C]
            identity (Tensor): 残差连接 [B, N_pv, C]
            query_pos (Tensor): 查询位置编码
            key_padding_mask (Tensor): 键填充掩码
            reference_points (Tensor): 参考点 [B, N_pv, D, 2]
            spatial_shapes (Tensor): 空间形状 [num_levels, 2]
            level_start_index (Tensor): 层级起始索引 [num_levels]
            bev_mask (Tensor): BEV掩码 [B, D, N_pv]，用于reference_points_rebatch
            
        Returns:
            Tensor: 注意力输出 [B, N_pv, C]
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        
        batch_size, num_query, _ = query.shape
        _, num_value, _ = value.shape
        
        # 验证空间形状和值数量匹配
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        
        # 值投影
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        
        value = value.view(batch_size, num_value, self.num_heads, -1)
        
        # 预测采样偏移量
        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        
        # 预测注意力权重
        attention_weights = self.attention_weights(query).view(
            batch_size, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(batch_size, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        # 处理参考点
        if reference_points.shape[-1] == 2:
            """
            对于每个PV查询，它在3D空间中有`num_Z_anchors`个不同高度的点。
            投影后，每个PV查询在每个BEV特征图上有`num_Z_anchors`个参考点。
            对于每个参考点，我们采样`num_points`个采样点。
            对于`num_Z_anchors`个参考点，总共有`num_points * num_Z_anchors`个采样点。
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)
        
        else:
            raise ValueError(
                f'Last dim of reference_points must be 2, but got {reference_points.shape[-1]} instead.')
        
        # 应用可变形注意力
        # 使用PyTorch实现，避免缺失的CUDA模块
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        
        # 输出投影
        output = self.output_proj(output)
        
        return output
