import torch
import torch.nn as nn
import time
import numpy as np
from functools import wraps
import os
import psutil

class PerformanceAnalyzer:
    """性能分析器，用于统计模块时间和参数量"""
    
    def __init__(self):
        self.timings = {}
        self.parameter_counts = {}
        self.current_iter = 0
        self.enabled = True
        
    def reset(self):
        """重置统计"""
        self.timings = {}
        self.parameter_counts = {}
        self.current_iter = 0
        
    def enable(self):
        """启用分析"""
        self.enabled = True
        
    def disable(self):
        """禁用分析"""
        self.enabled = False
        
    def time_module(self, module_name):
        """装饰器：用于统计模块执行时间"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                if module_name not in self.timings:
                    self.timings[module_name] = []
                self.timings[module_name].append(execution_time)
                
                return result
            return wrapper
        return decorator
    
    def count_parameters(self, model: nn.Module, module_name: str):
        """统计模块参数量"""
        if not self.enabled:
            return
            
        total_params = 0
        trainable_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        self.parameter_counts[module_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6
        }
    
    def get_memory_usage(self):
        """获取内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        else:
            allocated = reserved = max_allocated = 0
            
        process = psutil.Process(os.getpid())
        system_memory = process.memory_info().rss / 1024**3  # GB
        
        return {
            'gpu_allocated_gb': allocated,
            'gpu_reserved_gb': reserved,
            'gpu_max_allocated_gb': max_allocated,
            'system_memory_gb': system_memory
        }
    
    def print_summary(self, iteration: int = None):
        """打印性能摘要"""
        if not self.enabled:
            return
            
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        if iteration is not None:
            print(f"Iteration: {iteration}")
        
        # 打印参数量统计
        print("\n--- PARAMETER COUNTS ---")
        for module_name, counts in self.parameter_counts.items():
            print(f"{module_name:30s}: {counts['total_params_M']:8.2f}M total, "
                  f"{counts['trainable_params_M']:8.2f}M trainable")
        
        # 打印时间统计
        print("\n--- EXECUTION TIMES (seconds) ---")
        total_time = 0
        for module_name, times in self.timings.items():
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                total_time += avg_time
                print(f"{module_name:30s}: {avg_time:8.4f} ± {std_time:8.4f} "
                      f"(n={len(times)})")
        
        print(f"\nTotal average time: {total_time:.4f}s")
        
        # 打印内存使用
        memory_info = self.get_memory_usage()
        print("\n--- MEMORY USAGE ---")
        print(f"GPU Allocated: {memory_info['gpu_allocated_gb']:.2f} GB")
        print(f"GPU Reserved: {memory_info['gpu_reserved_gb']:.2f} GB")
        print(f"GPU Max Allocated: {memory_info['gpu_max_allocated_gb']:.2f} GB")
        print(f"System Memory: {memory_info['system_memory_gb']:.2f} GB")
        
        print("="*80 + "\n")
    
    def save_to_file(self, filename: str):
        """保存分析结果到文件"""
        if not self.enabled:
            return
            
        import json
        import datetime
        
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameter_counts': self.parameter_counts,
            'timings': {k: {'mean': float(np.mean(v)), 
                           'std': float(np.std(v)), 
                           'samples': len(v)} 
                       for k, v in self.timings.items()},
            'memory_usage': self.get_memory_usage()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Performance analysis saved to {filename}")


# 全局分析器实例
analyzer = PerformanceAnalyzer()

# 便捷装饰器函数
def time_module(module_name):
    return analyzer.time_module(module_name)

# 便捷统计函数
def count_parameters(model, module_name):
    analyzer.count_parameters(model, module_name)

def print_performance_summary(iteration=None):
    analyzer.print_summary(iteration)

def save_performance_results(filename):
    analyzer.save_to_file(filename)

def reset_performance_stats():
    analyzer.reset()

def enable_performance_analysis():
    analyzer.enable()

def disable_performance_analysis():
    analyzer.disable()
