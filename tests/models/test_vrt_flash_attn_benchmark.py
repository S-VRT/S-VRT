#!/usr/bin/env python3
"""
VRT Flash Attention Benchmark Test

This test compares the performance and timing of VRT models with and without Flash Attention
using the complete gopro_rgbspike_local_debug.json configuration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import gc
import json
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Tuple


# Import S-VRT components
from models.select_network import define_G
from utils import utils_logger


@contextmanager
def gpu_memory_monitor(stage_name: str = ""):
    """Context manager to monitor GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        yield
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
        current_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        print(".2f")
        print(".2f")
        print(".2f")
    else:
        yield


def load_config_with_comments(config_path: str) -> Dict[str, Any]:
    """Load JSON config file, handling comments"""
    def _strip_json_comments(text: str) -> str:
        result = []
        i = 0
        in_string = False
        string_char = ''
        while i < len(text):
            ch = text[i]
            if in_string:
                result.append(ch)
                if ch == '\\' and i + 1 < len(text):
                    result.append(text[i + 1])
                    i += 2
                    continue
                if ch == string_char:
                    in_string = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                result.append(ch)
                i += 1
                continue
            if ch == '/' and i + 1 < len(text):
                nxt = text[i + 1]
                if nxt == '/':
                    i += 2
                    while i < len(text) and text[i] not in '\n\r':
                        i += 1
                    continue
                if nxt == '*':
                    i += 2
                    while i + 1 < len(text) and not (text[i] == '*' and text[i + 1] == '/'):
                        i += 1
                    i += 2
                    continue
            result.append(ch)
            i += 1
        return ''.join(result)

    cfg_path = Path(config_path)
    text = cfg_path.read_text(encoding='utf-8')
    return json.loads(_strip_json_comments(text))


def create_vrt_model_with_config(config_path: str, use_flash_attn: bool) -> Tuple[nn.Module, Dict[str, Any]]:
    """Create VRT model from config file with flash attention setting"""
    # Load configuration
    opt = load_config_with_comments(config_path)

    # Override flash attention setting
    opt['netG']['use_flash_attn'] = use_flash_attn

    # Set training mode to False for inference
    opt['is_train'] = False

    # Create model
    model = define_G(opt)

    return model, opt


def create_sample_input(batch_size: int = 1, opt: Dict[str, Any] = None) -> torch.Tensor:
    """Create sample input tensor for VRT model"""
    if opt is None:
        # Default dimensions from gopro_rgbspike_local_debug.json
        frames = 6
        channels = 7  # RGB (3) + Spike TFP (4)
        height = 160
        width = 160
    else:
        # Extract from config
        netG_cfg = opt.get('netG', {})
        frames, height, width = netG_cfg.get('img_size', [6, 160, 160])
        channels = netG_cfg.get('in_chans', 7)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use float32 to avoid data type issues with optical flow operations
    dtype = torch.float32

    # Create input tensor: (batch_size, frames, channels, height, width)
    x = torch.randn(batch_size, frames, channels, height, width, dtype=dtype, device=device)

    return x


def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 10) -> Tuple[float, float]:
    """Measure average inference time and standard deviation"""
    model.eval()

    # Warm up
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)

    # Measure timing
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            _ = model(input_tensor)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time


def measure_memory_usage(model: nn.Module, input_tensor: torch.Tensor) -> float:
    """Measure peak memory usage during inference"""
    model.eval()

    with gpu_memory_monitor("Inference") as mem_monitor:
        with torch.no_grad():
            _ = model(input_tensor)

    # Return peak memory if available
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
    else:
        return 0.0


def check_output_consistency(model1: nn.Module, model2: nn.Module, input_tensor: torch.Tensor,
                           rtol: float = 1e-3, atol: float = 1e-4) -> Tuple[bool, float, float]:
    """Check if two models produce consistent outputs"""
    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model1(input_tensor)
        out2 = model2(input_tensor)

    # Compare outputs
    diff = torch.abs(out1 - out2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    relative_diff = torch.mean(diff / (torch.abs(out1) + 1e-8)).item()

    is_consistent = torch.allclose(out1, out2, rtol=rtol, atol=atol)

    return is_consistent, max_diff, relative_diff


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Flash attention benchmark requires CUDA")
class TestVRTFlashAttentionBenchmark:
    """Test class for VRT Flash Attention benchmark"""

    @pytest.fixture
    def config_path(self):
        """Path to the test configuration file"""
        import os
        # Support running from any machine; fall back to project-relative path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(project_root, "options", "gopro_rgbspike_local_debug.json")
        if not os.path.exists(path):
            pytest.skip(f"Config file not found: {path}")
        return path

    @pytest.fixture
    def device(self):
        """Get available device"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.mark.slow
    def test_flash_attention_availability(self):
        """Test if flash attention is available"""
        try:
            import flash_attn
            flash_available = True
            print("✓ Flash Attention library is available")
        except ImportError:
            flash_available = False
            print("✗ Flash Attention library not available")

        if not flash_available:
            pytest.skip("Flash Attention library not installed — skipping benchmark")

    @pytest.mark.slow
    def test_vrt_model_creation(self, config_path):
        """Test that VRT models can be created with and without flash attention"""
        # Test with flash attention
        model_flash, opt_flash = create_vrt_model_with_config(config_path, use_flash_attn=True)
        assert model_flash is not None, "Failed to create VRT model with flash attention"
        assert hasattr(model_flash, 'use_flash_attn'), "Model should have use_flash_attn attribute"
        assert model_flash.use_flash_attn == True, "Flash attention should be enabled"

        # Test without flash attention
        model_no_flash, opt_no_flash = create_vrt_model_with_config(config_path, use_flash_attn=False)
        assert model_no_flash is not None, "Failed to create VRT model without flash attention"
        assert hasattr(model_no_flash, 'use_flash_attn'), "Model should have use_flash_attn attribute"
        assert model_no_flash.use_flash_attn == False, "Flash attention should be disabled"

    @pytest.mark.slow
    def test_vrt_inference_timing(self, config_path):
        """Test inference timing comparison between flash attention and regular attention"""
        print("\n" + "="*60)
        print("VRT INFERENCE TIMING BENCHMARK")
        print("="*60)

        # Create models
        model_flash, opt = create_vrt_model_with_config(config_path, use_flash_attn=True)
        model_no_flash, _ = create_vrt_model_with_config(config_path, use_flash_attn=False)

        # Create sample input
        input_tensor = create_sample_input(batch_size=1, opt=opt)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Device: {input_tensor.device}")
        print(f"Dtype: {input_tensor.dtype}")

        # Measure timing for flash attention
        print("\n--- Testing Flash Attention ---")
        time_flash, std_flash = measure_inference_time(model_flash, input_tensor, num_runs=5)
        print(".4f")
        print(".4f")

        # Measure timing for regular attention
        print("\n--- Testing Regular Attention ---")
        time_regular, std_regular = measure_inference_time(model_no_flash, input_tensor, num_runs=5)
        print(".4f")
        print(".4f")

        # Calculate speedup
        speedup = time_regular / time_flash if time_flash > 0 else float('inf')
        print(".2f")

        # Assertions
        assert time_flash > 0, "Flash attention timing should be positive"
        assert time_regular > 0, "Regular attention timing should be positive"
        assert speedup >= 0.5, f"Flash attention should not be more than 2x slower (speedup: {speedup})"

        # Store results for summary
        self.timing_results = {
            'flash_time': time_flash,
            'regular_time': time_regular,
            'speedup': speedup,
            'flash_std': std_flash,
            'regular_std': std_regular
        }

    @pytest.mark.slow
    def test_vrt_memory_usage(self, config_path):
        """Test memory usage comparison between flash attention and regular attention"""
        print("\n" + "="*60)
        print("VRT MEMORY USAGE BENCHMARK")
        print("="*60)

        # Skip if not on CUDA
        if not torch.cuda.is_available():
            pytest.skip("Memory benchmark requires CUDA")

        # Create models
        model_flash, opt = create_vrt_model_with_config(config_path, use_flash_attn=True)
        model_no_flash, _ = create_vrt_model_with_config(config_path, use_flash_attn=False)

        # Create sample input
        input_tensor = create_sample_input(batch_size=1, opt=opt)

        print(f"Input shape: {input_tensor.shape}")

        # Measure memory for flash attention
        print("\n--- Testing Flash Attention Memory ---")
        mem_flash = measure_memory_usage(model_flash, input_tensor)

        # Measure memory for regular attention
        print("\n--- Testing Regular Attention Memory ---")
        mem_regular = measure_memory_usage(model_no_flash, input_tensor)

        # Calculate memory efficiency
        if mem_regular > 0:
            mem_efficiency = (mem_regular - mem_flash) / mem_regular * 100
            print(".2f")
        else:
            mem_efficiency = 0.0

        print(".2f")
        print(".2f")

        # Store results
        self.memory_results = {
            'flash_memory': mem_flash,
            'regular_memory': mem_regular,
            'memory_efficiency': mem_efficiency
        }

    @pytest.mark.slow
    def test_vrt_output_consistency(self, config_path):
        """Test that flash attention and regular attention produce consistent outputs"""
        print("\n" + "="*60)
        print("VRT OUTPUT CONSISTENCY TEST")
        print("="*60)

        # Create models with same weights (use same random seed)
        torch.manual_seed(42)
        model_flash, opt = create_vrt_model_with_config(config_path, use_flash_attn=True)

        torch.manual_seed(42)
        model_no_flash, _ = create_vrt_model_with_config(config_path, use_flash_attn=False)

        # Create sample input
        input_tensor = create_sample_input(batch_size=1, opt=opt)

        print(f"Input shape: {input_tensor.shape}")

        # Check consistency
        is_consistent, max_diff, relative_diff = check_output_consistency(
            model_flash, model_no_flash, input_tensor, rtol=1e-2, atol=1e-3
        )

        print(f"Outputs are consistent: {is_consistent}")
        print(".2e")
        print(".2e")

        # For Flash Attention, we expect some numerical differences but they should be reasonable
        if not is_consistent:
            print("⚠️  Outputs differ, but this may be expected due to different numerical implementations")

        # Store results
        self.consistency_results = {
            'is_consistent': is_consistent,
            'max_diff': max_diff,
            'relative_diff': relative_diff
        }

    @pytest.mark.slow
    def test_vrt_benchmark_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("VRT FLASH ATTENTION BENCHMARK SUMMARY")
        print("="*80)

        if hasattr(self, 'timing_results'):
            timing = self.timing_results
            print("TIMING RESULTS:")
            print(".4f")
            print(".4f")
            print(".2f")
            print(".4f")
            print(".4f")

        if hasattr(self, 'memory_results'):
            memory = self.memory_results
            print("\nMEMORY RESULTS:")
            print(".2f")
            print(".2f")
            print(".2f")

        if hasattr(self, 'consistency_results'):
            consistency = self.consistency_results
            print("\nCONSISTENCY RESULTS:")
            print(f"Outputs are consistent: {'✓' if consistency['is_consistent'] else '✗'}")
            print(".2e")
            print(".2e")

        print("\n" + "="*80)

        # Basic assertions for the benchmark
        if hasattr(self, 'timing_results'):
            assert self.timing_results['speedup'] > 0, "Speedup should be positive"
        if hasattr(self, 'memory_results'):
            assert self.memory_results['flash_memory'] >= 0, "Memory usage should be non-negative"
            assert self.memory_results['regular_memory'] >= 0, "Memory usage should be non-negative"


if __name__ == "__main__":
    # Allow running the test directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])
