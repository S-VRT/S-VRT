#!/usr/bin/env python3
"""
硬件资源对训练速度的影响分析工具

这个工具会：
1. 检测系统硬件资源（GPU、CPU、内存）
2. 分析GPU显存使用情况（包括CUDA缓存）
3. 运行轻量级基准测试
4. 分析训练配置的可行性
5. 解析训练日志并分析性能瓶颈（可选）
6. 提供优化建议和推荐配置

使用场景：
- 克隆项目后首次运行
- 更换硬件环境
- 训练配置优化
- 训练过程中的性能分析
"""

import os
import sys
import yaml
import time
import torch
import psutil
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 添加路径 - 支持从任意位置运行
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "VRT"))


class HardwareAnalyzer:
    """硬件资源分析器"""
    
    def __init__(self):
        self.gpu_info = []
        self.cpu_count = 0
        self.ram_gb = 0
        self.benchmark_results = {}
        
    def detect_hardware(self):
        """检测硬件配置"""
        print("=" * 80)
        print("硬件资源检测".center(80))
        print("=" * 80)
        
        # GPU检测
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"\n✅ 检测到 {num_gpus} 个GPU:")
            
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                gpu_name = props.name
                gpu_memory_gb = props.total_memory / (1024 ** 3)
                compute_capability = f"{props.major}.{props.minor}"
                
                self.gpu_info.append({
                    'index': i,
                    'name': gpu_name,
                    'memory_gb': gpu_memory_gb,
                    'compute_capability': compute_capability,
                    'multi_processor_count': props.multi_processor_count
                })
                
                print(f"  GPU {i}: {gpu_name}")
                print(f"    显存: {gpu_memory_gb:.1f} GB")
                print(f"    计算能力: {compute_capability}")
                print(f"    SM数量: {props.multi_processor_count}")
        else:
            print("\n❌ 未检测到可用的GPU")
            print("   本项目需要GPU进行训练")
            return False
        
        # CPU检测
        self.cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        print(f"\n💻 CPU信息:")
        print(f"  逻辑核心数: {self.cpu_count}")
        if cpu_freq:
            print(f"  当前频率: {cpu_freq.current:.0f} MHz")
            print(f"  最大频率: {cpu_freq.max:.0f} MHz")
        
        # 内存检测
        mem = psutil.virtual_memory()
        self.ram_gb = mem.total / (1024 ** 3)
        print(f"\n🧠 系统内存:")
        print(f"  总内存: {self.ram_gb:.1f} GB")
        print(f"  可用内存: {mem.available / (1024 ** 3):.1f} GB")
        print(f"  使用率: {mem.percent:.1f}%")
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        print(f"\n💾 磁盘空间:")
        print(f"  总空间: {disk.total / (1024 ** 3):.1f} GB")
        print(f"  可用空间: {disk.free / (1024 ** 3):.1f} GB")
        print(f"  使用率: {disk.percent:.1f}%")
        
        return True
    
    def analyze_gpu_memory_details(self):
        """详细分析GPU显存使用情况（包括CUDA缓存）"""
        print("\n" + "=" * 80)
        print("GPU显存详细分析".center(80))
        print("=" * 80)
        
        if not torch.cuda.is_available():
            print("\n❌ CUDA不可用，跳过显存分析")
            return
        
        # nvidia-smi视角的显存使用
        print("\n📊 nvidia-smi 显存使用情况:")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    idx, used, total = line.split(', ')
                    used_gb = int(used) / 1024
                    total_gb = int(total) / 1024
                    pct = (int(used) / int(total)) * 100
                    print(f"  GPU {idx}: {used_gb:.2f} GB / {total_gb:.2f} GB ({pct:.1f}%)")
        except Exception as e:
            print(f"  ⚠️  无法获取nvidia-smi信息: {e}")
        
        # PyTorch视角的显存使用
        print("\n📊 PyTorch 显存跟踪:")
        for i in range(len(self.gpu_info)):
            print(f"\n  GPU {i}:")
            torch.cuda.reset_peak_memory_stats(i)
            
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
            max_reserved = torch.cuda.max_memory_reserved(i) / 1024**3
            
            print(f"    已分配 (活跃tensor): {allocated:.2f} GB")
            print(f"    已保留 (PyTorch缓存): {reserved:.2f} GB")
            print(f"    峰值已分配: {max_allocated:.2f} GB")
            print(f"    峰值已保留: {max_reserved:.2f} GB")
            
            if reserved > allocated + 0.5:
                cache_gb = reserved - allocated
                print(f"    💡 CUDA缓存: {cache_gb:.2f} GB")
                print(f"       (这是PyTorch为性能优化保留的内存池)")
        
        # 查找GPU进程
        print("\n📊 当前GPU进程:")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(', ')
                        if len(parts) == 3:
                            pid, name, mem = parts
                            mem_gb = int(mem.replace(' MiB', '')) / 1024
                            print(f"  PID {pid}: {name} - {mem_gb:.2f} GB")
            else:
                print("  (无活跃进程)")
        except Exception as e:
            print(f"  ⚠️  无法获取进程信息: {e}")
        
        # 显存使用建议
        print("\n💡 显存优化提示:")
        print("  • PyTorch使用CUDA缓存分配器来提升性能")
        print("  • 'nvidia-smi'显示的是总保留内存(reserved)")
        print("  • 'torch.cuda.memory_allocated()'显示的是实际使用内存")
        print("  • 差值是PyTorch的内存缓存池，用于快速分配tensor")
        print("  • 如果遇到OOM，可以尝试:")
        print("    1. 启用混合精度训练(AMP) - 减少40-50%显存")
        print("    2. 使用gradient checkpointing")
        print("    3. 减小batch size")
        print("    4. 使用torch.cuda.empty_cache()清理缓存(临时方案)")
    
    def run_gpu_benchmark(self):
        """运行GPU基准测试"""
        print("\n" + "=" * 80)
        print("GPU性能基准测试".center(80))
        print("=" * 80)
        print("\n⏱️  正在运行轻量级基准测试（约30秒）...")
        
        if not self.gpu_info:
            print("❌ 无GPU可用，跳过基准测试")
            return
        
        # 对每个GPU进行测试
        for gpu in self.gpu_info:
            gpu_id = gpu['index']
            print(f"\n测试 GPU {gpu_id}: {gpu['name']}")
            
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            
            # 测试1: 矩阵乘法性能 (FP32)
            print("  [1/4] FP32矩阵乘法...")
            size = 4096
            a = torch.randn(size, size, device=f'cuda:{gpu_id}')
            b = torch.randn(size, size, device=f'cuda:{gpu_id}')
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            fp32_time = (time.time() - start) / 10
            fp32_tflops = (2 * size ** 3) / (fp32_time * 1e12)
            
            # 测试2: FP16性能
            print("  [2/4] FP16矩阵乘法...")
            a_fp16 = a.half()
            b_fp16 = b.half()
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                c_fp16 = torch.matmul(a_fp16, b_fp16)
            torch.cuda.synchronize()
            fp16_time = (time.time() - start) / 10
            fp16_tflops = (2 * size ** 3) / (fp16_time * 1e12)
            
            # 测试3: 显存带宽
            print("  [3/4] 显存带宽测试...")
            size_mb = 1024  # 1GB
            data = torch.randn(size_mb * 1024 * 1024 // 4, device=f'cuda:{gpu_id}')
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                data = data * 2.0
            torch.cuda.synchronize()
            bandwidth_time = (time.time() - start) / 10
            bandwidth_gb_s = (size_mb / 1024) / bandwidth_time
            
            # 测试4: 小批量推理延迟
            print("  [4/4] 推理延迟测试...")
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, padding=1),
                torch.nn.ReLU()
            ).to(f'cuda:{gpu_id}')
            
            test_input = torch.randn(1, 3, 256, 256, device=f'cuda:{gpu_id}')
            
            # Warmup
            for _ in range(5):
                _ = model(test_input)
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(50):
                _ = model(test_input)
            torch.cuda.synchronize()
            inference_time = (time.time() - start) / 50 * 1000  # ms
            
            # 保存结果
            self.benchmark_results[gpu_id] = {
                'fp32_tflops': fp32_tflops,
                'fp16_tflops': fp16_tflops,
                'bandwidth_gb_s': bandwidth_gb_s,
                'inference_latency_ms': inference_time,
                'fp16_speedup': fp16_tflops / fp32_tflops if fp32_tflops > 0 else 0
            }
            
            # 清理
            del a, b, c, a_fp16, b_fp16, c_fp16, data, model, test_input
            torch.cuda.empty_cache()
        
        # 打印结果
        print("\n" + "-" * 80)
        print("基准测试结果:")
        print("-" * 80)
        
        for gpu_id, results in self.benchmark_results.items():
            print(f"\nGPU {gpu_id}:")
            print(f"  FP32性能: {results['fp32_tflops']:.2f} TFLOPS")
            print(f"  FP16性能: {results['fp16_tflops']:.2f} TFLOPS")
            print(f"  FP16加速比: {results['fp16_speedup']:.2f}x")
            print(f"  显存带宽: {results['bandwidth_gb_s']:.1f} GB/s")
            print(f"  推理延迟: {results['inference_latency_ms']:.2f} ms")
    
    def estimate_training_requirements(self, config_path: str) -> Dict:
        """估算训练资源需求"""
        print("\n" + "=" * 80)
        print("训练资源需求分析".center(80))
        print("=" * 80)
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        batch_size = config['TRAIN']['BATCH_SIZE']
        gradient_accum = config['TRAIN']['GRADIENT_ACCUMULATION_STEPS']
        crop_size = config['DATA']['CROP_SIZE']
        clip_len = config['DATA']['CLIP_LEN']
        
        # 数据加载配置
        total_workers = config['DATALOADER']['TOTAL_WORKERS']
        cache_size_gb = config['DATA']['CACHE_SIZE_GB']
        
        print(f"\n📋 当前配置:")
        print(f"  Batch Size (per GPU): {batch_size}")
        print(f"  Gradient Accumulation: {gradient_accum}")
        print(f"  Effective Batch Size: {batch_size * gradient_accum * len(self.gpu_info)}")
        print(f"  Crop Size: {crop_size}x{crop_size}")
        print(f"  Clip Length: {clip_len}")
        print(f"  DataLoader Workers: {total_workers}")
        print(f"  Cache Size: {cache_size_gb} GB per worker")
        
        # 估算GPU显存需求
        print(f"\n🎮 GPU显存需求估算:")
        
        # 基础显存占用（粗略估算）
        model_params_gb = 5.0  # VRT base + Spike fusion模型约5GB
        optimizer_states_gb = model_params_gb * 2  # AdamW需要2倍参数量
        
        # 前向传播激活值（与batch size和分辨率相关）
        # 估算公式: batch * clip * channels * H * W * 4 (FP32) * layers
        activation_gb = batch_size * clip_len * 128 * crop_size * crop_size * 4 / (1024**3) * 8
        
        # 梯度（约等于模型参数）
        gradient_gb = model_params_gb
        
        # PyTorch CUDA缓存（经验值）
        cuda_cache_gb = 3.0
        
        total_gpu_memory_fp32 = (model_params_gb + optimizer_states_gb + 
                                  activation_gb + gradient_gb + cuda_cache_gb)
        
        # FP16混合精度可以减少约40-50%显存
        total_gpu_memory_fp16 = total_gpu_memory_fp32 * 0.55
        
        print(f"  模型参数: ~{model_params_gb:.1f} GB")
        print(f"  优化器状态: ~{optimizer_states_gb:.1f} GB")
        print(f"  激活值 (batch={batch_size}): ~{activation_gb:.1f} GB")
        print(f"  梯度: ~{gradient_gb:.1f} GB")
        print(f"  CUDA缓存: ~{cuda_cache_gb:.1f} GB")
        print(f"  ---")
        print(f"  预计总需求 (FP32): ~{total_gpu_memory_fp32:.1f} GB")
        print(f"  预计总需求 (FP16): ~{total_gpu_memory_fp16:.1f} GB")
        
        # 估算CPU和内存需求
        print(f"\n💻 CPU/内存需求估算:")
        
        # 解析workers配置
        if isinstance(total_workers, str):
            if total_workers == "auto" or total_workers.startswith("cpu*"):
                if total_workers.startswith("cpu*"):
                    ratio = float(total_workers.replace("cpu*", ""))
                    workers_per_gpu = int(self.cpu_count * ratio / len(self.gpu_info))
                else:
                    workers_per_gpu = int(self.cpu_count * 0.8 / len(self.gpu_info))
            else:
                workers_per_gpu = int(total_workers) // len(self.gpu_info)
        else:
            workers_per_gpu = total_workers // len(self.gpu_info)
        
        total_workers_count = workers_per_gpu * len(self.gpu_info)
        
        # 内存需求 = cache * workers + dataset overhead
        ram_for_cache = cache_size_gb * total_workers_count
        ram_overhead = 5.0  # 系统和其他开销
        total_ram_needed = ram_for_cache + ram_overhead
        
        print(f"  推荐Workers数: {workers_per_gpu} per GPU (总计 {total_workers_count})")
        print(f"  Cache占用: {cache_size_gb} GB/worker × {total_workers_count} = {ram_for_cache:.1f} GB")
        print(f"  系统开销: ~{ram_overhead:.1f} GB")
        print(f"  预计总需求: ~{total_ram_needed:.1f} GB")
        
        return {
            'batch_size': batch_size,
            'gradient_accum': gradient_accum,
            'gpu_memory_fp32': total_gpu_memory_fp32,
            'gpu_memory_fp16': total_gpu_memory_fp16,
            'ram_needed': total_ram_needed,
            'workers_per_gpu': workers_per_gpu,
            'total_workers': total_workers_count
        }
    
    def generate_recommendations(self, requirements: Dict):
        """生成配置优化建议"""
        print("\n" + "=" * 80)
        print("配置优化建议".center(80))
        print("=" * 80)
        
        recommendations = []
        warnings = []
        optimal_config = {}
        
        # 检查GPU显存
        min_gpu_memory = min([g['memory_gb'] for g in self.gpu_info])
        
        print(f"\n🎮 GPU配置建议:")
        
        if min_gpu_memory < requirements['gpu_memory_fp32']:
            if min_gpu_memory >= requirements['gpu_memory_fp16']:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'GPU',
                    'title': '启用混合精度训练 (AMP)',
                    'reason': f"当前GPU显存 ({min_gpu_memory:.1f} GB) 不足以支持FP32训练 ({requirements['gpu_memory_fp32']:.1f} GB)",
                    'action': '在配置中添加: TRAIN.USE_AMP: true',
                    'expected_benefit': '显存减少约45%, 训练速度提升20-30%',
                    'config_change': {'TRAIN': {'USE_AMP': True}}
                })
            else:
                warnings.append(f"⚠️  GPU显存可能不足: {min_gpu_memory:.1f} GB < {requirements['gpu_memory_fp16']:.1f} GB (FP16)")
                recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'GPU',
                    'title': '减小Batch Size',
                    'reason': f"GPU显存严重不足",
                    'action': f'将BATCH_SIZE从{requirements["batch_size"]}减小到1，或增加GRADIENT_ACCUMULATION_STEPS',
                    'expected_benefit': '确保训练可以运行',
                    'config_change': {'TRAIN': {'BATCH_SIZE': 1, 'GRADIENT_ACCUMULATION_STEPS': requirements['gradient_accum'] * requirements['batch_size']}}
                })
        else:
            # GPU显存充足，可以考虑增大batch size
            available_memory = min_gpu_memory - requirements['gpu_memory_fp32']
            if available_memory > 5:
                potential_batch_increase = int(available_memory / (requirements['gpu_memory_fp32'] / requirements['batch_size'] * 0.4))
                if potential_batch_increase > 1:
                    recommendations.append({
                        'priority': 'MEDIUM',
                        'category': 'GPU',
                        'title': '可以增大Batch Size',
                        'reason': f"GPU显存充足 ({min_gpu_memory:.1f} GB)，有 {available_memory:.1f} GB空闲",
                        'action': f'可以尝试将BATCH_SIZE增加到 {requirements["batch_size"] + potential_batch_increase}',
                        'expected_benefit': '训练更稳定，可能收敛更快',
                        'config_change': {'TRAIN': {'BATCH_SIZE': requirements["batch_size"] + potential_batch_increase}}
                    })
        
        # 检查FP16性能
        if self.benchmark_results:
            avg_speedup = sum([r['fp16_speedup'] for r in self.benchmark_results.values()]) / len(self.benchmark_results)
            if avg_speedup > 1.5:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'GPU',
                    'title': '启用混合精度训练以提升速度',
                    'reason': f"GPU的FP16性能是FP32的{avg_speedup:.1f}倍",
                    'action': '在配置中添加: TRAIN.USE_AMP: true',
                    'expected_benefit': f'训练速度提升约{(avg_speedup - 1) * 100:.0f}%',
                    'config_change': {'TRAIN': {'USE_AMP': True}}
                })
        
        # 检查CPU和内存
        print(f"\n💻 CPU/内存配置建议:")
        
        if self.ram_gb < requirements['ram_needed']:
            warnings.append(f"⚠️  系统内存可能不足: {self.ram_gb:.1f} GB < {requirements['ram_needed']:.1f} GB")
            
            # 计算合适的worker数和cache size
            safe_cache_size = (self.ram_gb * 0.7) / requirements['total_workers']
            safe_workers_ratio = (self.ram_gb * 0.7) / (requirements['total_workers'] * requirements['ram_needed'] / self.ram_gb)
            
            recommendations.append({
                'priority': 'HIGH',
                'category': 'CPU/RAM',
                'title': '减少DataLoader内存占用',
                'reason': f"系统内存不足 ({self.ram_gb:.1f} GB < {requirements['ram_needed']:.1f} GB)",
                'action': f'减少CACHE_SIZE_GB到{safe_cache_size:.1f}或减少workers',
                'expected_benefit': '避免OOM，确保稳定运行',
                'config_change': {
                    'DATA': {'CACHE_SIZE_GB': max(0.5, safe_cache_size)},
                    'DATALOADER': {'TOTAL_WORKERS': f"cpu*{max(0.3, safe_workers_ratio * 0.6):.1f}"}
                }
            })
        else:
            available_ram = self.ram_gb - requirements['ram_needed']
            if available_ram > 20:
                recommendations.append({
                    'priority': 'LOW',
                    'category': 'CPU/RAM',
                    'title': '可以增加Cache Size',
                    'reason': f"系统内存充足 ({self.ram_gb:.1f} GB)，有 {available_ram:.1f} GB空闲",
                    'action': f'可以增加CACHE_SIZE_GB到 {requirements["ram_needed"] / requirements["total_workers"] + 1:.1f}',
                    'expected_benefit': '提升数据加载速度',
                    'config_change': {'DATA': {'CACHE_SIZE_GB': min(3.0, requirements["ram_needed"] / requirements["total_workers"] + 1)}}
                })
        
        # CPU cores检查
        optimal_workers = int(self.cpu_count * 0.7 / len(self.gpu_info))
        if requirements['workers_per_gpu'] > self.cpu_count:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'CPU/RAM',
                'title': '调整DataLoader Workers数量',
                'reason': f"Workers数({requirements['total_workers']})超过CPU核心数({self.cpu_count})",
                'action': f'设置TOTAL_WORKERS为"cpu*0.7" (约{optimal_workers * len(self.gpu_info)}个workers)',
                'expected_benefit': '避免CPU过载',
                'config_change': {'DATALOADER': {'TOTAL_WORKERS': 'cpu*0.7'}}
            })
        
        # 打印建议
        if warnings:
            print("\n⚠️  警告:")
            for warning in warnings:
                print(f"  {warning}")
        
        print("\n📊 优化建议 (按优先级排序):\n")
        
        # 按优先级排序
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        for i, rec in enumerate(recommendations, 1):
            priority_icon = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(rec['priority'], '⚪')
            
            print(f"{priority_icon} [{rec['priority']}] {rec['title']}")
            print(f"   分类: {rec['category']}")
            print(f"   原因: {rec['reason']}")
            print(f"   建议: {rec['action']}")
            print(f"   预期效果: {rec['expected_benefit']}")
            print()
        
        return recommendations
    
    def generate_optimized_config(self, config_path: str, recommendations: List[Dict], output_path: str):
        """生成优化后的配置文件"""
        print("=" * 80)
        print("生成优化配置".center(80))
        print("=" * 80)
        
        # 加载原始配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 应用高优先级建议
        applied_changes = []
        for rec in recommendations:
            if rec['priority'] in ['CRITICAL', 'HIGH']:
                if 'config_change' in rec:
                    for section, changes in rec['config_change'].items():
                        if section not in config:
                            config[section] = {}
                        config[section].update(changes)
                    applied_changes.append(rec['title'])
        
        # 保存优化后的配置
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ 已生成优化配置: {output_file}")
        print(f"\n已应用的优化:")
        for change in applied_changes:
            print(f"  ✓ {change}")
        
        print(f"\n💡 使用方法:")
        print(f"  python src/train.py --config {output_file}")
        
        return output_file
    
    def analyze_training_log(self, log_file: Path):
        """分析训练日志中的性能数据"""
        print("\n" + "=" * 80)
        print("训练日志性能分析".center(80))
        print("=" * 80)
        
        if not log_file.exists():
            print(f"\n⚠️  日志文件不存在: {log_file}")
            return
        
        print(f"\n📄 分析日志文件: {log_file.name}\n")
        
        # 解析日志
        stage_times = defaultdict(list)
        fusion_times = defaultdict(list)
        forward_times = []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 提取总前向传播时间
                if match := re.search(r'前向传播总耗时: ([\d.]+)ms', line):
                    forward_times.append(float(match.group(1)))
                
                # 提取各Stage耗时
                if match := re.search(r'\[VRT\] Stage (\d+).*?耗时: ([\d.]+)ms', line):
                    stage_num = int(match.group(1))
                    time_ms = float(match.group(2))
                    stage_times[stage_num].append(time_ms)
                
                # 提取融合耗时
                if match := re.search(r'\[VRT融合\] Stage (\d+) 总耗时: ([\d.]+)ms', line):
                    stage_num = int(match.group(1))
                    time_ms = float(match.group(2))
                    fusion_times[stage_num].append(time_ms)
        
        if not forward_times:
            print("⚠️  日志中未找到性能计时信息")
            print("   提示: 确保训练时开启了timing调试输出")
            return
        
        # 总体性能统计
        print("📊 总体性能:")
        avg_forward = sum(forward_times) / len(forward_times)
        print(f"  前向传播平均耗时: {avg_forward:.2f} ms")
        print(f"  最小/最大: {min(forward_times):.2f} / {max(forward_times):.2f} ms")
        print(f"  样本数: {len(forward_times)}")
        
        # 各Stage性能分析
        if stage_times:
            print("\n📊 VRT各Stage性能分析:")
            print("-" * 80)
            
            stage_data = []
            for stage_num in sorted(stage_times.keys()):
                times = stage_times[stage_num]
                if times:
                    avg_time = sum(times) / len(times)
                    percentage = (avg_time / avg_forward) * 100
                    stage_data.append((stage_num, avg_time, percentage))
                    
                    stage_name = {
                        1: "Stage 1 (1x分辨率)",
                        2: "Stage 2 (1/2x分辨率)",
                        3: "Stage 3 (1/4x分辨率)",
                        4: "Stage 4 (1/8x分辨率)",
                        5: "Stage 5 (瓶颈层)",
                        6: "Stage 6 (解码1/4x)",
                        7: "Stage 7 (解码1/2x)",
                        8: "Stage 8 (重建层)",
                    }.get(stage_num, f"Stage {stage_num}")
                    
                    # 根据占比标记优先级
                    if percentage > 15:
                        priority = "🔴 极高"
                    elif percentage > 8:
                        priority = "🟠 高"
                    elif percentage > 5:
                        priority = "🟡 中"
                    else:
                        priority = "✅ 低"
                    
                    print(f"\n  {stage_name}")
                    print(f"    平均耗时: {avg_time:.2f} ms ({percentage:.1f}%)")
                    print(f"    优化优先级: {priority}")
            
            # 性能瓶颈排名
            print("\n" + "-" * 80)
            print("🔥 性能瓶颈排名 (Top 5):")
            stage_data.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (stage_num, avg_time, percentage) in enumerate(stage_data[:5], 1):
                stage_name = {
                    1: "Stage 1 (1x)", 2: "Stage 2 (1/2x)", 3: "Stage 3 (1/4x)",
                    4: "Stage 4 (1/8x)", 5: "Stage 5 (瓶颈)", 6: "Stage 6 (解码1/4x)",
                    7: "Stage 7 (解码1/2x)", 8: "Stage 8 (重建)",
                }.get(stage_num, f"Stage {stage_num}")
                
                print(f"  #{rank}. {stage_name}: {avg_time:.2f} ms ({percentage:.1f}%)")
            
            # 优化建议
            print("\n💡 性能优化建议:")
            if stage_data and stage_data[0][2] > 30:
                print(f"  • Stage {stage_data[0][0]} 是主要瓶颈，占用{stage_data[0][2]:.1f}%时间")
                print(f"    建议优先优化该阶段")
            
            high_cost_stages = [s for s in stage_data if s[2] > 15]
            if len(high_cost_stages) >= 2:
                print(f"  • 发现{len(high_cost_stages)}个高耗时阶段(>15%)")
                print(f"    建议考虑模型架构优化")
            
            print(f"  • 通用优化:")
            print(f"    - 启用混合精度训练(AMP)")
            print(f"    - 使用Flash Attention(如果支持)")
            print(f"    - 减少Transformer块数量")
    
    def estimate_training_speed(self, requirements: Dict):
        """估算训练速度"""
        print("\n" + "=" * 80)
        print("训练速度预估".center(80))
        print("=" * 80)
        
        if not self.benchmark_results:
            print("\n⚠️  未运行基准测试，无法估算训练速度")
            return
        
        # 基于基准测试结果估算
        avg_inference_ms = sum([r['inference_latency_ms'] for r in self.benchmark_results.values()]) / len(self.benchmark_results)
        
        # VRT模型比简单CNN复杂约30-50倍（粗略估算）
        estimated_forward_ms = avg_inference_ms * 40
        
        # backward pass约为forward的2倍
        estimated_backward_ms = estimated_forward_ms * 2
        
        # 每个训练step的总时间
        step_time_ms = estimated_forward_ms + estimated_backward_ms
        
        # 考虑梯度累积
        effective_step_time_ms = step_time_ms * requirements['gradient_accum']
        
        # 数据加载时间（假设与计算并行，取较大值）
        data_loading_ms = 50  # 假设数据加载50ms
        
        total_step_time_s = max(effective_step_time_ms / 1000, data_loading_ms / 1000)
        
        steps_per_hour = 3600 / total_step_time_s
        
        print(f"\n⏱️  预估训练速度:")
        print(f"  单次前向传播: ~{estimated_forward_ms:.0f} ms")
        print(f"  单次反向传播: ~{estimated_backward_ms:.0f} ms")
        print(f"  每个step (含梯度累积): ~{total_step_time_s:.1f} s")
        print(f"  训练速度: ~{steps_per_hour:.0f} steps/hour")
        
        # 估算完整训练时间
        typical_steps = 300000  # 典型的总训练步数
        estimated_hours = typical_steps / steps_per_hour
        estimated_days = estimated_hours / 24
        
        print(f"\n📅 预估完整训练时间 ({typical_steps:,} steps):")
        print(f"  总计: {estimated_hours:.1f} 小时 ({estimated_days:.1f} 天)")
        
        print(f"\n💡 加速建议:")
        print(f"  • 使用混合精度训练 (AMP) 可提速 20-30%")
        print(f"  • 增大batch size可提高GPU利用率")
        print(f"  • 使用多GPU可线性加速")


def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("硬件资源与训练速度分析工具".center(80))
    print("=" * 80)
    print("\n本工具将帮助你:")
    print("  1. 检测系统硬件资源")
    print("  2. 分析GPU显存使用情况")
    print("  3. 运行性能基准测试")
    print("  4. 分析训练配置的可行性")
    print("  5. 分析训练日志性能(可选)")
    print("  6. 生成优化建议和推荐配置")
    print("\n")
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='硬件资源与训练速度分析工具')
    parser.add_argument('--config', type=str, default='configs/deblur/vrt_spike_baseline.yaml',
                       help='配置文件路径')
    parser.add_argument('--log', type=str, default=None,
                       help='训练日志文件路径(可选，用于性能分析)')
    parser.add_argument('--auto', action='store_true',
                       help='自动模式：跳过所有交互式询问')
    
    # 兼容旧的命令行方式
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        args = argparse.Namespace(config=sys.argv[1], log=None, auto=False)
    else:
        args = parser.parse_args()
    
    # 处理配置文件路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        print(f"\n用法: python {Path(__file__).name} [--config CONFIG] [--log LOG] [--auto]")
        print(f"示例: python {Path(__file__).name} --config configs/deblur/vrt_spike_baseline.yaml")
        return
    
    print(f"📄 分析配置文件: {config_path.relative_to(REPO_ROOT)}\n")
    
    # 创建分析器
    analyzer = HardwareAnalyzer()
    
    # 1. 检测硬件
    if not analyzer.detect_hardware():
        print("\n❌ 硬件检测失败，请确保系统有可用的GPU")
        return
    
    # 2. GPU显存详细分析
    if args.auto:
        run_memory_analysis = True
    else:
        run_memory_analysis = input("\n是否运行GPU显存详细分析? [Y/n]: ").strip().lower() != 'n'
    
    if run_memory_analysis:
        analyzer.analyze_gpu_memory_details()
    
    # 3. 运行基准测试
    if args.auto:
        run_benchmark = True
    else:
        run_benchmark = input("\n是否运行GPU基准测试? (约30秒) [Y/n]: ").strip().lower() != 'n'
    
    if run_benchmark:
        analyzer.run_gpu_benchmark()
    
    # 4. 分析训练需求
    requirements = analyzer.estimate_training_requirements(str(config_path))
    
    # 5. 生成建议
    recommendations = analyzer.generate_recommendations(requirements)
    
    # 6. 估算训练速度
    if run_benchmark:
        analyzer.estimate_training_speed(requirements)
    
    # 7. 分析训练日志(如果提供)
    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = REPO_ROOT / log_path
        analyzer.analyze_training_log(log_path)
    else:
        # 尝试自动查找最新的训练日志
        log_dir = REPO_ROOT / "outputs" / "logs"
        if log_dir.exists():
            log_files = sorted(log_dir.glob("train_*.log"), 
                             key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                if args.auto:
                    analyze_log = True
                else:
                    analyze_log = input(f"\n发现训练日志 {log_files[0].name}，是否分析? [y/N]: ").strip().lower() == 'y'
                
                if analyze_log:
                    analyzer.analyze_training_log(log_files[0])
    
    # 8. 生成优化配置
    if recommendations:
        if args.auto:
            generate_config = True
        else:
            generate_config = input("\n是否生成优化后的配置文件? [Y/n]: ").strip().lower() != 'n'
        
        if generate_config:
            output_name = config_path.stem + "_optimized.yaml"
            output_path = config_path.parent / output_name
            analyzer.generate_optimized_config(
                str(config_path),
                recommendations,
                str(output_path)
            )
    
    # 总结
    print("\n" + "=" * 80)
    print("分析完成".center(80))
    print("=" * 80)
    print("\n📝 下一步:")
    print("  1. 查看上述优化建议")
    print("  2. 根据建议修改配置文件")
    print("  3. 运行训练前测试: python tests/integration/training/test_system_readiness.py")
    print("  4. 开始训练: python src/train.py --config <your_config>.yaml")
    print("\n💡 提示:")
    print("  • 首次训练建议使用较小的batch size和较短的验证间隔")
    print("  • 训练过程中可使用 nvidia-smi 监控GPU使用情况")
    print("  • 遇到OOM错误时，尝试减小batch size或启用gradient checkpointing")
    print("  • 可以使用 --auto 参数运行自动模式，跳过所有询问")
    print("\n")


if __name__ == "__main__":
    main()

