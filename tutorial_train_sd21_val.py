import sys
sys.path.append("./scripts")
import cv2
from share import *
import torch.distributed as dist
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
import torch
import numpy as np
# 新增：导入检查点回调
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    # 设置内存优化配置
    torch.backends.cuda.max_split_size_mb = 128  # 避免内存碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    # 启用xFormers以优化性能
    os.environ['XFORMERS_DISABLED'] = '0'

    # Configs
    resume_path = './models/control_sd21_ini.ckpt'
    batch_size = 1  # 每个GPU的batch_size
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # 实验名称（用于区分不同训练任务的保存路径）
    experiment_name = "251108"
    # 检查点保存根目录
    checkpoint_root = os.path.join("./experiments", experiment_name, "checkpoints")
    os.makedirs(checkpoint_root, exist_ok=True)  # 确保目录存在

    # 使用CPU加载模型，然后手动移动到GPU确保所有组件都在正确设备上
    print("Loading model on CPU...")
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # 数据加载器配置 - 减少工作进程数以避免内存竞争
    train_dataset = MyDataset("./data_tmp/train_prompt.json")
    val_dataset = MyDataset("./data_tmp/val_prompt.json")
    
    train_dataloader = DataLoader(train_dataset, num_workers=64, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=64, batch_size=batch_size, shuffle=False)
    
    logger_loader = DataLoader(val_dataset, num_workers=64, batch_size=1, shuffle=False)
    logger_data = next(iter(logger_loader))

    # 日志记录器（图像日志）

    logger = ImageLogger(batch_frequency=logger_freq, logger_data=logger_data)

    # 配置检查点回调（保存模型权重）
    # 1. 保存最新模型
    last_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_root,
        filename="last-epoch={epoch:02d}-step={step:06d}",  # 文件名包含epoch和step
        save_last=True,  # 始终保存最新的检查点
        save_weights_only=False,  # 只保存模型权重（减小体积）
        verbose=True
    )

    # 2. 保存验证损失最低的最优模型
    best_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_root,
        filename="best-epoch={epoch:02d}-step={step:06d}",  # 包含损失值
        monitor="val/loss",  # 监控验证集损失
        mode="min",  # 最小化损失
        save_top_k=1,  # 只保存最优的1个模型
        save_weights_only=False,
        verbose=True
    )

    # 训练器配置 - 整合所有回调（图像日志+检查点）
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        strategy='ddp',
        precision=32,  # 若显存不足，可改为16（混合精度训练）
        callbacks=[logger, last_ckpt_callback, best_ckpt_callback],  # 包含最优模型回调
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        max_epochs=1000,
        check_val_every_n_epoch=1,  # 每个epoch都进行验证
        val_check_interval=1.0,     # 每个epoch结束后验证
        log_every_n_steps=100
    )

    print("Starting training with multi-GPU support...")
    # print(f"Training samples: {train_size}, Validation samples: {val_size}")
    print(f"Total effective batch size: {batch_size * 4}")  # batch_size * num_gpus
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 训练！同时提供训练和验证数据加载器
    trainer.fit(model, train_dataloader, val_dataloader)

    # 训练完成后打印最优模型路径
    print(f"Best model saved at: {best_ckpt_callback.best_model_path}")
    print(f"Best validation loss: {best_ckpt_callback.best_model_score:.4f}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':

    main()
