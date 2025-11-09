**环境配置**
python==3.8
cuda版本切换到11.7 配xformers==0.0.23 PyTorch==1.13.1
[从官网下载CLIP模型放入：models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin]
[从官网下载SD冻结模型放入：models/v2-1_512-ema-pruned.ckpt]

**构建模型**
cd scripts
python tool_add_control_sd21.py /mnt/mydisk/fkx/CVPR/baseline/models/v2-1_512-ema-pruned.ckpt /mnt/mydisk/fkx/CVPR/control_revised/models/control_sd21_ini.ckpt

**数据配置**
生成数据读取路径：training_data/infrared2color/prompt.json
改datalodaer: scripts/tutorial_dataset.py

**训练**
改模型文件：cldm/cldm.py
改采样文件：ldm/models/diffusion/ddpm.py与ldm/models/diffusion/ddim.py
改训练文件：tutorial_train_sd21.py

训练：python tutorial_train_sd21.py