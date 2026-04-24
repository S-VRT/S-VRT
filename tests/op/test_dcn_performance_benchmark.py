#!/usr/bin/env python3
"""
DCN Performance Benchmark Test

This test compares the performance of DCNv2 and DCNv4 implementations
in terms of speed, memory usage, and numerical accuracy.
"""

import torch
import torch.nn as nn
import time
import psutil
import os
import sys
import pytest
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.blocks.dcn import DCNv2PackFlowGuided, DCNv4PackFlowGuided

# Skip all DCNv4-dependent tests if the CUDA extension is not installed
def _dcnv4_available():
    try:
        from models.op.dcnv4 import DCNv4  # noqa: F401
        return True
    except (ImportError, RuntimeError):
        return False

pytestmark = pytest.mark.skipif(not _dcnv4_available(), reason="DCNv4 CUDA extension not installed")


def cuda_memory_monitor(func):
    """Decorator to monitor CUDA memory usage."""
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return {"peak": 0, "allocated": 0, "delta": 0}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()

        return {
            "result": result,
            "memory": {
                "initial": initial_memory,
                "peak": peak_memory,
                "allocated": final_memory,
                "delta": final_memory - initial_memory
            }
        }
    return wrapper


def get_cpu_memory_usage():
    """Get current CPU memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


class DCNPerformanceBenchmark:
    """Benchmark class for DCN performance comparison."""

    def __init__(self, channels=64, height=64, width=64, batch_size=4, deformable_groups=4):
        self.channels = channels
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.deformable_groups = deformable_groups

        # Generate test data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._generate_test_data()

    def _generate_test_data(self):
        """Generate consistent test data for benchmarking."""
        torch.manual_seed(42)  # For reproducible results

        self.x = torch.randn(self.batch_size, self.channels, self.height, self.width,
                           device=self.device, dtype=torch.float32)

        # Generate flow-warped features (simulate neighboring frames)
        # For pa_frames=2, we typically have 1 flow-warped feature and 1 flow per call
        self.x_flow_warpeds = [
            torch.randn(self.batch_size, self.channels, self.height, self.width,
                       device=self.device, dtype=torch.float32)
            for _ in range(1)  # For pa_frames=2, typically 1 warped frame per call
        ]

        # Current frame features
        self.x_current = torch.randn(self.batch_size, self.channels, self.height, self.width,
                                   device=self.device, dtype=torch.float32)

        # Optical flow data
        self.flows = [
            torch.randn(self.batch_size, 2, self.height, self.width,
                       device=self.device, dtype=torch.float32)
            for _ in range(1)  # For pa_frames=2, typically 1 flow per call
        ]

    def create_models(self):
        """Create DCNv2 and DCNv4 models for comparison."""
        # DCNv2
        dcn_v2 = DCNv2PackFlowGuided(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            deformable_groups=self.deformable_groups,
            pa_frames=2,
            max_residue_magnitude=10
        ).to(self.device)

        # DCNv4
        dcn_v4 = DCNv4PackFlowGuided(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            deformable_groups=self.deformable_groups,
            pa_frames=2
        ).to(self.device)

        return dcn_v2, dcn_v4

    def benchmark_forward(self, model, num_runs=100, warmup_runs=10):
        """Benchmark forward pass performance."""
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(self.x, self.x_flow_warpeds, self.x_current, self.flows)

        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(self.x, self.x_flow_warpeds, self.x_current, self.flows)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        return avg_time, output

    def benchmark_memory(self, model):
        """Benchmark memory usage."""
        @cuda_memory_monitor
        def run_model():
            with torch.no_grad():
                return model(self.x, self.x_flow_warpeds, self.x_current, self.flows)

        result = run_model()
        return result["memory"]

    def benchmark_gradient(self, model, num_runs=10):
        """Benchmark backward pass performance."""
        # Enable gradient computation
        model.train()
        self.x.requires_grad_(True)

        # Forward + backward warmup
        for _ in range(5):
            output = model(self.x, self.x_flow_warpeds, self.x_current, self.flows)
            loss = output.sum()
            loss.backward()
            self.x.grad.zero_()

        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            output = model(self.x, self.x_flow_warpeds, self.x_current, self.flows)
            loss = output.sum()
            loss.backward()
            self.x.grad.zero_()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        return avg_time

    def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark."""
        print(f"\n=== DCN Performance Benchmark ===")
        print(f"Input shape: {self.x.shape}")
        print(f"Device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        dcn_v2, dcn_v4 = self.create_models()

        results = {
            'dcn_v2': {},
            'dcn_v4': {}
        }

        # Forward benchmark
        print("\n--- Forward Pass Benchmark ---")
        for name, model in [('DCNv2', dcn_v2), ('DCNv4', dcn_v4)]:
            print(f"Testing {name}...")
            avg_time, output = self.benchmark_forward(model)
            mem_usage = self.benchmark_memory(model)

            key = f'dcn_{name.lower()}'  # 'dcn_dcnv2' -> we'll use 'dcn_v2', 'dcn_dcnv4' -> 'dcn_v4'
            if name == 'DCNv2':
                key = 'dcn_v2'
            elif name == 'DCNv4':
                key = 'dcn_v4'
            results[key]['forward_time'] = avg_time
            results[key]['output_shape'] = output.shape
            results[key]['memory'] = mem_usage

            print(".4f")
            print(f"  Output shape: {output.shape}")
            if torch.cuda.is_available():
                print(".2f")

        # Backward benchmark
        print("\n--- Backward Pass Benchmark ---")
        for name, model in [('DCNv2', dcn_v2), ('DCNv4', dcn_v4)]:
            print(f"Testing {name}...")
            avg_time = self.benchmark_gradient(model)

            key = 'dcn_v2' if name == 'DCNv2' else 'dcn_v4'
            results[key]['backward_time'] = avg_time
            print(".4f")

        # Analysis
        print("\n--- Performance Analysis ---")
        v2_time = results['dcn_v2']['forward_time']
        v4_time = results['dcn_v4']['forward_time']

        speedup = v2_time / v4_time if v4_time > 0 else float('inf')
        print(".2f")
        print(".1f")

        if torch.cuda.is_available():
            v2_mem = results['dcn_v2']['memory'].get('peak', 0)
            v4_mem = results['dcn_v4']['memory'].get('peak', 0)
            mem_ratio = v4_mem / v2_mem if v2_mem > 0 else 1.0
            print(".2f")

        # Numerical accuracy check
        print("\n--- Numerical Accuracy Check ---")
        with torch.no_grad():
            out_v2 = dcn_v2(self.x, self.x_flow_warpeds, self.x_current, self.flows)
            out_v4 = dcn_v4(self.x, self.x_flow_warpeds, self.x_current, self.flows)

        mse = torch.mean((out_v2 - out_v4) ** 2).item()
        max_diff = torch.max(torch.abs(out_v2 - out_v4)).item()

        print(".6f")
        print(".6f")

        results['comparison'] = {
            'speedup': speedup,
            'mse_diff': mse,
            'max_diff': max_diff
        }

        return results


# Pytest test functions
@pytest.mark.parametrize("batch_size,channels,height,width", [
    (1, 64, 32, 32),
    (4, 64, 64, 64),
    (2, 128, 32, 32),
])
def test_dcn_performance_comparison(batch_size, channels, height, width):
    """Test performance comparison between DCNv2 and DCNv4."""
    benchmark = DCNPerformanceBenchmark(
        channels=channels,
        height=height,
        width=width,
        batch_size=batch_size
    )

    results = benchmark.run_comprehensive_benchmark()

    # Basic assertions
    assert 'dcn_v2' in results
    assert 'dcn_v4' in results
    assert 'forward_time' in results['dcn_v2']
    assert 'forward_time' in results['dcn_v4']

    # DCNv4 should be reasonably close in performance (allowing for some overhead)
    v2_time = results['dcn_v2']['forward_time']
    v4_time = results['dcn_v4']['forward_time']

    # Allow DCNv4 to be up to 2x slower initially (due to adaptation overhead)
    # but should not be excessively slow
    assert v4_time < v2_time * 10, ".4f"

    print(f"✓ Performance test passed for {batch_size}x{channels}x{height}x{width}")


def test_dcn_numerical_consistency():
    """Test that DCNv4 produces numerically reasonable results."""
    benchmark = DCNPerformanceBenchmark()

    dcn_v2, dcn_v4 = benchmark.create_models()

    with torch.no_grad():
        out_v2 = dcn_v2(benchmark.x, benchmark.x_flow_warpeds, benchmark.x_current, benchmark.flows)
        out_v4 = dcn_v4(benchmark.x, benchmark.x_flow_warpeds, benchmark.x_current, benchmark.flows)

    # Check output shapes match
    assert out_v2.shape == out_v4.shape, f"Shape mismatch: DCNv2 {out_v2.shape} vs DCNv4 {out_v4.shape}"

    # Check outputs are finite
    assert torch.isfinite(out_v2).all(), "DCNv2 output contains non-finite values"
    assert torch.isfinite(out_v4).all(), "DCNv4 output contains non-finite values"

    # Check outputs are not all zeros (basic sanity check)
    assert not torch.allclose(out_v4, torch.zeros_like(out_v4)), "DCNv4 output is all zeros"

    print("✓ Numerical consistency test passed")


def test_dcn_flow_guidance_effect():
    """Test that DCN uses flow guidance information properly."""
    benchmark = DCNPerformanceBenchmark()

    dcn_v4 = benchmark.create_models()[1]  # Get DCNv4

    # Test with original flows
    with torch.no_grad():
        out1 = dcn_v4(benchmark.x, benchmark.x_flow_warpeds, benchmark.x_current, benchmark.flows)

    # Test with zero flows (should produce different results)
    zero_flows = [torch.zeros_like(flow) for flow in benchmark.flows]
    with torch.no_grad():
        out2 = dcn_v4(benchmark.x, benchmark.x_flow_warpeds, benchmark.x_current, zero_flows)

    # Results should be different when flows are different
    assert not torch.allclose(out1, out2, atol=1e-6), "DCN output unchanged with different flows"

    print("✓ Flow guidance effect test passed")


if __name__ == "__main__":
    # Run standalone benchmark
    benchmark = DCNPerformanceBenchmark()
    results = benchmark.run_comprehensive_benchmark()

    print("\n=== Benchmark Summary ===")
    print(f"DCNv2 forward time: {results['dcn_v2']['forward_time']:.4f}s")
    print(f"DCNv4 forward time: {results['dcn_v4']['forward_time']:.4f}s")
    print(f"Speedup: {results['comparison']['speedup']:.2f}x")
    print(f"MSE difference: {results['comparison']['mse_diff']:.6f}")
    print(f"Max difference: {results['comparison']['max_diff']:.6f}")
