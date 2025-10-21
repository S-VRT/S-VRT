#!/usr/bin/env python3
"""
Training Speed Benchmark Script

This script benchmarks training speed with different configurations to validate optimizations.
"""

import argparse
import time
import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to Python path
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config
from models.build import build_model
from data.gopro_spike_dataset import build_dataloader


def measure_forward_backward(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    use_amp: bool = False,
    iterations: int = 10,
) -> Dict[str, float]:
    """Measure forward and backward pass times."""
    
    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Move batch to device
    blur = batch['blur'].to(device)
    sharp = batch['sharp'].to(device)
    spike_voxel = batch['spike_voxel'].to(device)
    
    # Warmup
    for _ in range(3):
        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(blur, spike_voxel)
            loss = nn.functional.mse_loss(output, sharp)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(torch.optim.SGD(model.parameters(), lr=0.001))
            scaler.update()
        else:
            loss.backward()
        
        model.zero_grad()
    
    # Benchmark
    torch.cuda.synchronize()
    
    forward_times = []
    backward_times = []
    total_times = []
    
    for _ in range(iterations):
        start = time.time()
        
        # Forward pass
        forward_start = time.time()
        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(blur, spike_voxel)
            loss = nn.functional.mse_loss(output, sharp)
        torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - backward_start
        
        total_time = time.time() - start
        
        forward_times.append(forward_time)
        backward_times.append(backward_time)
        total_times.append(total_time)
        
        model.zero_grad()
    
    return {
        'forward_mean': sum(forward_times) / len(forward_times),
        'forward_std': torch.tensor(forward_times).std().item(),
        'backward_mean': sum(backward_times) / len(backward_times),
        'backward_std': torch.tensor(backward_times).std().item(),
        'total_mean': sum(total_times) / len(total_times),
        'total_std': torch.tensor(total_times).std().item(),
    }


def measure_dataloader_speed(
    dataloader,
    iterations: int = 50,
) -> Dict[str, float]:
    """Measure dataloader speed."""
    
    times = []
    
    # Warmup
    for i, _ in enumerate(dataloader):
        if i >= 3:
            break
    
    # Benchmark
    start = time.time()
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        # Simulate some processing
        _ = batch['blur'].shape
        times.append(time.time() - batch_start)
        
        if i >= iterations - 1:
            break
    
    total_time = time.time() - start
    
    return {
        'batch_mean': sum(times) / len(times),
        'batch_std': torch.tensor(times).std().item(),
        'throughput': iterations / total_time,  # batches/sec
    }


def print_results(
    config_name: str,
    model_results: Dict[str, float],
    dataloader_results: Dict[str, float],
    batch_size: int,
    num_gpus: int,
):
    """Pretty print benchmark results."""
    
    print(f"\n{'='*80}")
    print(f"  Benchmark Results: {config_name}")
    print(f"{'='*80}")
    
    print(f"\n📊 Model Performance:")
    print(f"  Forward pass:   {model_results['forward_mean']*1000:.1f} ± {model_results['forward_std']*1000:.1f} ms")
    print(f"  Backward pass:  {model_results['backward_mean']*1000:.1f} ± {model_results['backward_std']*1000:.1f} ms")
    print(f"  Total (F+B):    {model_results['total_mean']*1000:.1f} ± {model_results['total_std']*1000:.1f} ms")
    
    print(f"\n📦 DataLoader Performance:")
    print(f"  Batch load time: {dataloader_results['batch_mean']*1000:.1f} ± {dataloader_results['batch_std']*1000:.1f} ms")
    print(f"  Throughput:      {dataloader_results['throughput']:.2f} batches/sec")
    
    # Calculate training estimates
    step_time = model_results['total_mean']
    steps_per_hour = 3600 / step_time
    
    print(f"\n⏱️  Training Speed Estimates:")
    print(f"  Steps/hour:       {steps_per_hour:.0f}")
    print(f"  Samples/hour:     {steps_per_hour * batch_size * num_gpus:.0f}")
    print(f"  Time for 300K steps: {300000 / steps_per_hour:.1f} hours ({300000 / steps_per_hour / 24:.1f} days)")
    
    print(f"{'='*80}\n")


def benchmark_config(config_path: str, iterations: int = 10):
    """Benchmark a single configuration."""
    
    print(f"\n🔍 Loading config: {config_path}")
    cfg = load_config(config_path)
    
    # Get config info
    batch_size = cfg.get('TRAIN', {}).get('BATCH_SIZE', 1)
    use_amp = cfg.get('TRAIN', {}).get('USE_AMP', False)
    num_gpus = torch.cuda.device_count()
    
    print(f"   Batch size: {batch_size}")
    print(f"   AMP: {use_amp}")
    print(f"   GPUs: {num_gpus}")
    
    # Build model
    print(f"🏗️  Building model...")
    model = build_model(cfg)
    model = model.cuda()
    model.train()
    
    # Build dataloader
    print(f"📦 Building dataloader...")
    train_loader, _ = build_dataloader(cfg, rank=0, world_size=1)
    
    # Get a batch
    print(f"⏳ Fetching batch...")
    batch = next(iter(train_loader))
    
    # Benchmark model
    print(f"⚡ Benchmarking model ({iterations} iterations)...")
    model_results = measure_forward_backward(model, batch, use_amp=use_amp, iterations=iterations)
    
    # Benchmark dataloader
    print(f"⚡ Benchmarking dataloader ({iterations} batches)...")
    dataloader_results = measure_dataloader_speed(train_loader, iterations=iterations)
    
    # Print results
    print_results(
        Path(config_path).stem,
        model_results,
        dataloader_results,
        batch_size,
        num_gpus,
    )
    
    return {
        'model': model_results,
        'dataloader': dataloader_results,
        'batch_size': batch_size,
        'num_gpus': num_gpus,
    }


def compare_configs(baseline_path: str, optimized_path: str, iterations: int = 10):
    """Compare baseline and optimized configurations."""
    
    print(f"\n{'='*80}")
    print(f"  Training Speed Comparison")
    print(f"{'='*80}")
    
    # Benchmark baseline
    print(f"\n[1/2] Benchmarking BASELINE configuration...")
    baseline_results = benchmark_config(baseline_path, iterations)
    
    # Benchmark optimized
    print(f"\n[2/2] Benchmarking OPTIMIZED configuration...")
    optimized_results = benchmark_config(optimized_path, iterations)
    
    # Calculate improvements
    print(f"\n{'='*80}")
    print(f"  Performance Improvement Summary")
    print(f"{'='*80}")
    
    baseline_step = baseline_results['model']['total_mean']
    optimized_step = optimized_results['model']['total_mean']
    speedup = baseline_step / optimized_step
    
    baseline_throughput = baseline_results['dataloader']['throughput']
    optimized_throughput = optimized_results['dataloader']['throughput']
    dataloader_speedup = optimized_throughput / baseline_throughput
    
    print(f"\n🚀 Model Training:")
    print(f"   Baseline:   {baseline_step*1000:.1f} ms/step")
    print(f"   Optimized:  {optimized_step*1000:.1f} ms/step")
    print(f"   Speedup:    {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    
    print(f"\n📦 DataLoader:")
    print(f"   Baseline:   {baseline_throughput:.2f} batches/sec")
    print(f"   Optimized:  {optimized_throughput:.2f} batches/sec")
    print(f"   Speedup:    {dataloader_speedup:.2f}x ({(dataloader_speedup-1)*100:.1f}% faster)")
    
    baseline_hours = 300000 / (3600 / baseline_step)
    optimized_hours = 300000 / (3600 / optimized_step)
    time_saved = baseline_hours - optimized_hours
    
    print(f"\n⏰ Time for 300K steps:")
    print(f"   Baseline:   {baseline_hours:.1f} hours ({baseline_hours/24:.1f} days)")
    print(f"   Optimized:  {optimized_hours:.1f} hours ({optimized_hours/24:.1f} days)")
    print(f"   Time saved: {time_saved:.1f} hours ({time_saved/24:.1f} days)")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Benchmark training speed')
    parser.add_argument('--config', type=str, help='Single config to benchmark')
    parser.add_argument('--baseline', type=str, default='configs/deblur/vrt_spike_baseline.yaml',
                       help='Baseline config for comparison')
    parser.add_argument('--optimized', type=str, default='configs/deblur/vrt_spike_baseline_optimized.yaml',
                       help='Optimized config for comparison')
    parser.add_argument('--compare', action='store_true', help='Compare baseline vs optimized')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for benchmark')
    
    args = parser.parse_args()
    
    if args.config:
        benchmark_config(args.config, args.iterations)
    elif args.compare:
        compare_configs(args.baseline, args.optimized, args.iterations)
    else:
        # Default: just benchmark the optimized config
        benchmark_config(args.optimized, args.iterations)


if __name__ == '__main__':
    main()

