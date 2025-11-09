import torch
import sys
import os
import numpy as np

# 添加当前目录到路径，以便可以导入同目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 添加性能分析器
sys.path.append(os.path.dirname(current_dir))
from performance_analyzer import time_module, count_parameters

from builder import create_model

# 全局模型实例，避免重复创建
_sate_model = None

@time_module("sate_forward")
def sate_forward(images):
    global _sate_model
    
    if _sate_model is None:
        _sate_model = create_model('default')
        # 统计sate模型参数量
        count_parameters(_sate_model, "sate_model")
    
    # 确保模型与输入数据在同一个设备上
    device = images.device
    _sate_model = _sate_model.to(device)
    
    # 设置模型为评估模式
    # _sate_model.eval()
    
    # 创建模拟输入数据
    lidar_to_infra=np.array([552.763666521189, -1603.78477199458, -3.83219023979035, -1412558.53923310,
                        477.106478425300, -42.3540800995409, -1579.29515654294, -441861.107127547,
                        0.998131, -0.0494392, -0.0359335, -537.265,
                        0, 0, 0, 1]).reshape((4,4))
    lidar_to_infra_torch = torch.from_numpy(lidar_to_infra).float().to(device)  # 移动到相同设备

    # 3. 扩展batch维度（B=1），最终形状为 [1, 4, 4]
    world2pv_param = lidar_to_infra_torch.unsqueeze(0) #.repeat(4, 1, 1) # 增加batch维度
    
    camera_params = {'world2pv': world2pv_param}

    # with torch.no_grad():
    output = _sate_model(images, camera_params)
    return output['output']
        

if __name__ == "__main__":
    batch_size = 1
    satellite_img = torch.randn(batch_size, 3, 1000, 1000)  # 卫星图像 [B, 3, 500, 500]
    sate_forward(satellite_img)
