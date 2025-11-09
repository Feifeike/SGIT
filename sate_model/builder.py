import torch
import sys
sys.path.append("/mnt/mydisk/fkx/CVPR/control_revised/sate_model")
from mmengine.registry import Registry, build_from_cfg

from satellite2pv import Satellite2PV

# 创建模型注册表
SATE_MODELS = Registry('sate_model')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')


def build_sate_model(cfg):
    """构建Satellite2PV模型
    
    Args:
        cfg (dict): 模型配置
        
    Returns:
        nn.Module: 构建的模型
    """
    return build_from_cfg(cfg, SATE_MODELS)


def build_backbone(cfg):
    """构建主干网络
    
    Args:
        cfg (dict): 主干网络配置
        
    Returns:
        nn.Module: 构建的主干网络
    """
    return build_from_cfg(cfg, BACKBONES)

# 导入模型类（这些类已经在各自的文件中通过@MODELS.register_module()注册）


class ModelBuilder:
    """模型构建工具类"""
    
    @staticmethod
    def create_satellite2pv_model(config):
        """根据配置创建Satellite2PV模型
        
        Args:
            config (dict): 模型配置
            
        Returns:
            Satellite2PV: 创建的模型
        """
        
        # 提取配置参数
        satellite_encoder_cfg = config.get('satellite_encoder', {})
        inverse_projection_cfg = config.get('inverse_projection', {})
        
        # 创建模型
        model = Satellite2PV(
            satellite_encoder_cfg=satellite_encoder_cfg,
            inverse_projection_cfg=inverse_projection_cfg,
        )
        
        return model
    
    @staticmethod
    def create_default_model():
        """创建默认配置的Satellite2PV模型
        
        Returns:
            Satellite2PV: 默认模型
        """
        default_config = {
            'satellite_encoder': {
                'backbone_cfg': {
                    'type': 'ResNet',
                    'depth': 50,
                    'num_stages': 4,
                    'out_indices': (3,),
                    'frozen_stages': 1,
                    'norm_cfg': {'type': 'BN2d', 'requires_grad': False},
                    'norm_eval': True,
                    'style': 'caffe',
                    'with_cp': True,
                    'dcn': {'type': 'DCNv2', 'deform_groups': 1, 'fallback_on_stride': False},
                    'stage_with_dcn': (False, False, True, True)
                },
                'neck_cfg': {
                    'type': 'FPN',
                    'in_channels': [2048],
                    'out_channels': 320, 
                    'start_level': 0,
                    'add_extra_convs': 'on_output',
                    'num_outs': 1,
                    'relu_before_extra_convs': True
                }
            },
            'inverse_projection': {
                'hidden_channels': 320,
                'num_heads': 8,
                'num_layers': 2,
                'pv_size': [48,80,100],
                'bev_range': (0,-100,200,100)
            }
        }
        
        return ModelBuilder.create_satellite2pv_model(default_config)
    
    @staticmethod
    def create_multi_task_model():
        """创建多任务Satellite2PV模型
        
        Returns:
            Satellite2PV: 多任务模型
        """
        multi_task_config = {
            'satellite_encoder': {
                'backbone_cfg': {
                    'type': 'ResNet',
                    'depth': 50,
                    'num_stages': 4,
                    'out_indices': (3,),
                    'frozen_stages': 1,
                    'norm_cfg': {'type': 'BN2d', 'requires_grad': False},
                    'norm_eval': True,
                    'style': 'caffe',
                    'with_cp': True,
                    'dcn': {'type': 'DCNv2', 'deform_groups': 1, 'fallback_on_stride': False},
                    'stage_with_dcn': (False, False, True, True)
                },
                'neck_cfg': {
                    'type': 'FPN',
                    'in_channels': [2048],
                    'out_channels': 256,
                    'start_level': 0,
                    'add_extra_convs': 'on_output',
                    'num_outs': 1,
                    'relu_before_extra_convs': True
                }
            },
            'inverse_projection': {
                'in_channels': 256,
                'hidden_channels': 512,
                'num_heads': 8,
                'num_layers': 6
            },
            'bev_size': (500, 500),
            'pv_size': (1080, 1920)
        }
        
        return ModelBuilder.create_satellite2pv_model(multi_task_config)


# 便捷函数
def create_model(model_type='default', **kwargs):
    """便捷创建模型函数
    
    Args:
        model_type (str): 模型类型 ('default', 'multi_task', 'custom')
        **kwargs: 自定义参数
        
    Returns:
        Satellite2PV: 创建的模型
    """
    if model_type == 'default':
        return ModelBuilder.create_default_model()
    elif model_type == 'multi_task':
        return ModelBuilder.create_multi_task_model()
    elif model_type == 'custom':
        return ModelBuilder.create_satellite2pv_model(kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_model(model, checkpoint_path, strict=True):
    """加载预训练权重
    
    Args:
        model (nn.Module): 模型实例
        checkpoint_path (str): 权重文件路径
        strict (bool): 是否严格匹配
        
    Returns:
        nn.Module: 加载权重后的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 加载权重
    model.load_state_dict(state_dict, strict=strict)
    
    return model
