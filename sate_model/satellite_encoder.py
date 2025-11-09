from mmengine.model import BaseModule
from mmengine.registry import build_from_cfg
from mmdet.registry import MODELS   

class SatelliteEncoder(BaseModule):
    """基于ResNet+FPN的卫星地图编码器"""
    
    def __init__(self,
                 backbone_cfg=None,
                 neck_cfg=None,
                 init_cfg=None):
        super(SatelliteEncoder, self).__init__(init_cfg)
        
        # 默认ResNet50配置
        if backbone_cfg is None:
            backbone_cfg = dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3),  # 输出3阶段的特征
                frozen_stages=1,
                norm_cfg=dict(type='BN2d', requires_grad=False),
                norm_eval=True,
                style='caffe',
                with_cp=True,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True)
            )
        
        # 默认FPN配置
        if neck_cfg is None:
            neck_cfg = dict(
                type='FPN',
                in_channels=[2048],  # ResNet50各阶段输出通道
                out_channels=256,
                start_level=0,
                add_extra_convs='on_output',
                num_outs=1,  # 输出1个尺度的特征
                relu_before_extra_convs=True
            )
        

        self.backbone = build_from_cfg(backbone_cfg, MODELS)  # 使用注册表构建
        self.neck = build_from_cfg(neck_cfg, MODELS)  # FPN 也在 MODELS 注册表中

        # 获取输出通道数
        self.out_channels = neck_cfg['out_channels']
        
    
    def forward(self, x):
        """前向传播
        
        Args:
            x (Tensor): 输入卫星图像 [B, C, H, W]
            
        Returns:
            dict: 包含多尺度特征的字典
                - 'features': 多尺度特征列表 [P2, P3, P4, P5]
                - 'backbone_features': 原始主干特征
        """
        # 主干网络提取多尺度特征
        backbone_features = self.backbone(x)
        
        # FPN融合多尺度特征
        fpn_features = self.neck(backbone_features)
        
        return fpn_features
    
    def get_output_channels(self):
        """获取输出通道数"""
        return self.out_channels
    
    def get_feature_shapes(self, input_shape):
        """获取各层特征图尺寸
        
        Args:
            input_shape (tuple): 输入图像形状 (C, H, W)
            
        Returns:
            dict: 各层特征图尺寸
        """
        _, H, W = input_shape
        shapes = {
            'P2': (self.out_channels, H // 4, W // 4),    # 1/4
            'P3': (self.out_channels, H // 8, W // 8),    # 1/8  
            'P4': (self.out_channels, H // 16, W // 16),  # 1/16
            'P5': (self.out_channels, H // 32, W // 32),  # 1/32
        }
        return shapes
