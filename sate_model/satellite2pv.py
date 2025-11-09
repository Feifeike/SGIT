from mmengine.model import BaseModule
from mmdet.registry import MODELS
import sys
sys.path.append("/mnt/mydisk/fkx/CVPR/control_revised/sate_model")
from satellite_encoder import SatelliteEncoder
from inverse_projection_transformer import InverseProjectionTransformer


@MODELS.register_module()
class Satellite2PV(BaseModule):
    """Satellite to Perspective View (PV) Model
    
    将卫星地图BEV特征转换为目标视角的透视视图(PV)图像
    """
    
    def __init__(self,
                 satellite_encoder_cfg,
                 inverse_projection_cfg,
                 init_cfg=None):
        super(Satellite2PV, self).__init__(init_cfg)
    
        
        # 卫星地图编码器
        self.satellite_encoder = SatelliteEncoder(**satellite_encoder_cfg)
        
        # 逆投影变换器
        self.inverse_projection = InverseProjectionTransformer(
            **inverse_projection_cfg)
    
    def forward(self, satellite_img, camera_params, targets=None):
        """前向传播
        
        Args:
            satellite_img (Tensor): 卫星地图图像 [B, 3, H, W]
            camera_params (dict): 相机参数，包含内外参
            targets (dict, optional): 训练时的目标数据
            
        Returns:
            dict: 包含预测结果和损失
        """
        
        # 1. 卫星地图特征提取
        fpn_features = self.satellite_encoder(satellite_img)
        
        # 提取BEV特征（使用FPN输出的第一个特征图）
        if isinstance(fpn_features, (list, tuple)):
            bev_features = fpn_features[0]  # 使用第一个尺度的特征
        else:
            bev_features = fpn_features
        
        # 2. 逆投影变换到PV空间
        pv_features = self.inverse_projection(bev_features, camera_params)
        
        return {'output': pv_features}
