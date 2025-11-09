import sys
sys.path.append("./scripts")

from share import *
import torch.distributed as dist
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os
import torch

from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    # 设置内存优化配置
    torch.backends.cuda.max_split_size_mb = 128  # 避免内存碎片
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Configs
    resume_path = './models/control_sd21_ini.ckpt'
    batch_size = 1  # 每个GPU的batch_size
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # 实验名称（用于区分不同训练任务的保存路径）
    experiment_name = "251028"
    # 检查点保存根目录
    checkpoint_root = os.path.join("./experiments", experiment_name, "checkpoints")
    os.makedirs(checkpoint_root, exist_ok=True)  # 确保目录存在

    # 使用CPU加载模型，然后移动到GPU
    print("Loading model on CPU...")
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # 数据加载器配置 - 减少工作进程数以避免内存竞争
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    # 日志记录器（图像日志）
    logger = ImageLogger(batch_frequency=logger_freq)

    # 新增：配置检查点回调（保存模型权重）
    # 1. 保存最新模型
    last_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_root,
        filename="last-epoch={epoch:02d}-step={step:06d}",  # 文件名包含epoch和step
        save_last=True,  # 始终保存最新的检查点
        save_weights_only=True,  # 只保存模型权重（减小体积）
        verbose=True
    )

    # 2. 保存验证损失最低的最优模型
    best_ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_root,
        filename="best-epoch={epoch:02d}-val_loss={val/loss:.4f}",  # 包含损失值
        monitor="val/loss",  # 监控验证集损失（需模型输出该指标）
        mode="min",  # 最小化损失
        save_top_k=1,  # 只保存最优的1个模型
        save_weights_only=True,
        verbose=True
    )

    # 训练器配置 - 整合所有回调（图像日志+检查点）
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy='ddp',
        precision=32,  # 若显存不足，可改为16（混合精度训练）
        callbacks=[logger, last_ckpt_callback], #best_ckpt_callback],  # 新增检查点回调
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        max_epochs=80,
        # check_val_every_n_epoch=10,  # 每10个epoch验证一次（用于更新最优模型）
        log_every_n_steps=100,
        # 若需要从断点续训，添加以下参数
        # resume_from_checkpoint=resume_path if resume_path else None
    )

    print("Starting training with multi-GPU support...")
    print(f"Total effective batch size: {batch_size * 4}")  # batch_size * num_gpus
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # 训练！
    trainer.fit(model, dataloader)

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()