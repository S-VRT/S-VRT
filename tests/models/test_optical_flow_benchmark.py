import torch
import time
import psutil
import os
from models.optical_flow import create_optical_flow


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def benchmark_optical_flow():
    """Benchmark memory usage and inference speed of optical flow models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmark on device: {device}")

    # Test configurations
    configs = [
        ('spynet', 'weights/optical_flow/spynet_sintel_final-3d2a1287.pth'),
        ('sea_raft', 'weights/optical_flow/Tartan-C-T-TSKH-kitti432x960-M.pth')
    ]

    results = {}

    for model_name, checkpoint in configs:
        print(f"\n=== Benchmarking {model_name} ===")

        # Load model
        try:
            model = create_optical_flow(model_name, checkpoint, device=device)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        # Test data
        batch_size = 4
        h, w = 256, 256
        frames1 = torch.rand(batch_size, 3, h, w, device=device)
        frames2 = torch.rand(batch_size, 3, h, w, device=device)

        # Warm up
        print("Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = model(frames1, frames2)

        # Memory before inference
        torch.cuda.empty_cache() if device == 'cuda' else None
        mem_before = get_memory_usage()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            mem_gpu_before = torch.cuda.memory_allocated() / 1024 / 1024

        # Time inference
        print("Running inference benchmark...")
        num_runs = 10
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model(frames1, frames2)
        end_time = time.time()

        # Memory after inference
        mem_after = get_memory_usage()
        if device == 'cuda':
            mem_gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_gpu_after = torch.cuda.memory_allocated() / 1024 / 1024

        # Calculate metrics
        avg_time = (end_time - start_time) / num_runs
        fps = 1.0 / avg_time

        # Check output shapes
        if isinstance(outputs, list):
            output_shapes = [out.shape for out in outputs]
        else:
            output_shapes = [outputs.shape]

        results[model_name] = {
            'avg_inference_time': avg_time,
            'fps': fps,
            'cpu_mem_before': mem_before,
            'cpu_mem_after': mem_after,
            'cpu_mem_delta': mem_after - mem_before,
            'output_shapes': output_shapes
        }

        if device == 'cuda':
            results[model_name].update({
                'gpu_mem_peak': mem_gpu_peak,
                'gpu_mem_after': mem_gpu_after,
                'gpu_mem_delta': mem_gpu_peak - mem_gpu_before
            })

        print(".3f")
        print(f"CPU Memory: {mem_before:.1f}MB -> {mem_after:.1f}MB (Δ{mem_after - mem_before:.1f}MB)")
        if device == 'cuda':
            print(f"GPU Memory Peak: {mem_gpu_peak:.1f}MB (Δ{mem_gpu_peak - mem_gpu_before:.1f}MB)")
        print(f"Output shapes: {output_shapes}")

    # Compare results
    print("\n=== Comparison ===")
    if 'spynet' in results and 'sea_raft' in results:
        spynet = results['spynet']
        sea_raft = results['sea_raft']

        time_ratio = sea_raft['avg_inference_time'] / spynet['avg_inference_time']
        print(".2f")
        print(".1f")

        if device == 'cuda':
            gpu_ratio = sea_raft['gpu_mem_peak'] / spynet['gpu_mem_peak']
            print(".2f")

    return results


if __name__ == "__main__":
    benchmark_optical_flow()
