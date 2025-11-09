import torch
import time
from cldm.model import create_model
from performance_analyzer import count_parameters, time_module, print_performance_summary, enable_performance_analysis
from scripts.tutorial_dataset import MyDataset
from torch.utils.data import DataLoader

def test_accurate_performance():
    """更准确的模型性能测试，模拟真实训练流程"""
    print("=== 准确模型性能测试 ===")
    
    # 启用性能分析
    enable_performance_analysis()
    
    # 加载模型
    print("加载模型中...")
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(torch.load('./models/control_sd21_ini.ckpt', map_location='cpu'))
    model = model.cuda()
    
    # 统计参数量
    print("\n=== 参数量统计 ===")
    count_parameters(model.control_model, 'ControlNet')
    count_parameters(model.model.diffusion_model, 'ControlledUnetModel')
    count_parameters(model.first_stage_model, 'AutoencoderKL (VAE)')
    count_parameters(model.cond_stage_model, 'CLIP Text Encoder')
    
    # 加载数据集
    print("\n=== 加载数据集 ===")
    dataset = MyDataset("./data_tmp/train_prompt.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 获取多组数据用于更准确的测试
    print("加载多组数据...")
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= 10:  # 取10个样本进行测试
            break
        batches.append(batch)
    
    print(f"加载了 {len(batches)} 个样本用于测试")
    
    # 多次迭代测试性能 - 模拟完整训练流程
    num_iterations = 10  # 增加迭代次数以获得更稳定的平均值
    print(f"\n=== 进行 {num_iterations} 次迭代训练测试 ===")
    
    # 启用训练模式
    model.control_model.train()
    model.model.diffusion_model.train()
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.control_model.parameters(), lr=1e-5)
    
    total_times = []
    
    for i in range(num_iterations):
        print(f"迭代 {i+1}/{num_iterations}")
        
        # 使用不同的数据样本
        batch_idx = i % len(batches)
        batch = batches[batch_idx]
        
        # 提取真实数据并调整维度 (HWC -> CHW)
        target = batch["source"].permute(0, 3, 1, 2).cuda()
        seg = batch["seg"].permute(0, 3, 1, 2).cuda()
        depth = batch["depth"].permute(0, 3, 1, 2).cuda()
        infrared = batch["infrared"].permute(0, 3, 1, 2).cuda()
        sate = batch["sate"].permute(0, 3, 1, 2).cuda()
        prompt = batch["prompt"]
        
        # 准备控制信号
        control_signals = [seg, depth, infrared, sate]
        
        # 创建潜在空间输入 (4通道)
        with torch.no_grad():
            latent_input = model.encode_first_stage(target).mode()
        
        # 重置梯度
        optimizer.zero_grad()
        
        # 前向传播计时
        forward_start = time.time()
        
        control_output = model.control_model(
            x=latent_input,
            hint=control_signals,
            timesteps=torch.randint(0, 1000, (1,)).cuda(),
            context=torch.randn(1, 77, 1024).cuda()  # 模拟文本条件
        )
        
        # UNet前向传播
        predicted_noise = model.model.diffusion_model(
            x=latent_input,
            timesteps=torch.randint(0, 1000, (1,)).cuda(),
            context=torch.randn(1, 77, 1024).cuda(),
            control=control_output
        )
        
        # 计算损失函数 (模拟扩散模型损失)
        target_noise = torch.randn_like(predicted_noise)
        loss = torch.nn.functional.mse_loss(predicted_noise, target_noise)
        
        forward_end = time.time()
        forward_time = forward_end - forward_start
        
        # 反向传播计时
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        backward_time = backward_end - backward_start
        
        # 优化器更新
        optimizer.step()
        
        # 总迭代时间
        iter_time = forward_time + backward_time
        total_times.append(iter_time)
        
        # 记录前向和反向时间
        if i == 0:
            print(f"  前向传播时间: {forward_time:.4f}s")
            print(f"  反向传播时间: {backward_time:.4f}s")
            print(f"  总迭代时间: {iter_time:.4f}s")
        
        if (i + 1) % 10 == 0:
            avg_time = sum(total_times[-10:]) / 10
            print(f"  最近10次平均时间: {avg_time:.4f}s")
    
    # 打印性能摘要
    print_performance_summary()
    
    # 计算准确的时间估算
    print("\n=== 准确训练时间估算 ===")
    avg_iter_time = sum(total_times) / len(total_times)
    print(f"平均每次迭代时间: {avg_iter_time:.4f}s")
    
    # 重新估算训练时间
    calibrated_avg_time = avg_iter_time
    print(f"\n校准后的平均迭代时间: {calibrated_avg_time:.4f}s")
    
    # 假设训练1000个epoch，每个epoch有5035个样本，batch_size=1
    total_samples = 5035  # 根据您的实际数据
    total_epochs = 100
    total_iterations = total_samples * total_epochs
    
    estimated_total_time = calibrated_avg_time * total_iterations
    estimated_hours = estimated_total_time / 3600
    estimated_days = estimated_hours / 24
    
    print(f"\n=== 重新校准的训练时间估算 ===")
    print(f"总迭代次数: {total_iterations:,}")
    print(f"总时间: {estimated_total_time/3600:.2f} 小时 ({estimated_days:.2f} 天)")
    print(f"每个epoch时间: {calibrated_avg_time * total_samples / 60:.2f} 分钟")
    

    return calibrated_avg_time

if __name__ == "__main__":
    calibrated_time = test_accurate_performance()
