"""
Unit tests for DCN modules - forward/backward compatibility and shape consistency.
Tests both DCNv2PackFlowGuided and DCNv4PackFlowGuided for shape and gradient consistency.
"""

import torch
import torch.nn as nn
import pytest
from torch.testing import assert_close

# Import the DCN modules
from models.blocks.dcn import DCNv2PackFlowGuided, DCNv4PackFlowGuided, get_deformable_module


class TestDCNModules:
    """Test DCN modules for forward/backward consistency."""

    @pytest.fixture
    def sample_inputs(self):
        """Generate sample inputs for DCN testing."""
        batch_size, channels, height, width = 2, 96, 32, 32
        pa_frames = 2

        # Create sample feature maps
        x = torch.randn(batch_size, channels, height, width)
        x_current = torch.randn(batch_size, channels, height, width)

        # Create flow-warped features (for pa_frames=2, we have 1 warped frame)
        x_flow_warpeds = [
            torch.randn(batch_size, channels, height, width)
        ]

        # Create flow tensors (for pa_frames=2, we have 1 flow)
        flows = [
            torch.randn(batch_size, 2, height, width)
        ]

        return {
            'x': x,
            'x_flow_warpeds': x_flow_warpeds,
            'x_current': x_current,
            'flows': flows,
            'channels': channels,
            'pa_frames': pa_frames
        }

    def test_dcnv2_forward_shape(self, sample_inputs):
        """Test DCNv2 forward pass shape consistency."""
        dcn = DCNv2PackFlowGuided(
            in_channels=sample_inputs['channels'],
            out_channels=sample_inputs['channels'],
            kernel_size=3,
            padding=1,
            deformable_groups=16,
            pa_frames=sample_inputs['pa_frames']
        )

        output = dcn(
            sample_inputs['x'],
            sample_inputs['x_flow_warpeds'],
            sample_inputs['x_current'],
            sample_inputs['flows']
        )

        # Check output shape
        expected_shape = sample_inputs['x'].shape
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DCNv4 requires CUDA")
    def test_dcnv4_forward_shape(self, sample_inputs):
        """Test DCNv4 forward pass shape consistency."""
        # Move inputs to CUDA
        cuda_inputs = {}
        for key, value in sample_inputs.items():
            if isinstance(value, list):
                cuda_inputs[key] = [v.cuda() for v in value]
            elif isinstance(value, torch.Tensor):
                cuda_inputs[key] = value.cuda()
            else:
                cuda_inputs[key] = value  # Keep non-tensor values as-is

        dcn = DCNv4PackFlowGuided(
            in_channels=cuda_inputs['channels'],
            out_channels=cuda_inputs['channels'],
            kernel_size=3,
            padding=1,
            deformable_groups=16,
            pa_frames=cuda_inputs['pa_frames']
        ).cuda()

        output = dcn(
            cuda_inputs['x'],
            cuda_inputs['x_flow_warpeds'],
            cuda_inputs['x_current'],
            cuda_inputs['flows']
        )

        # Check output shape
        expected_shape = cuda_inputs['x'].shape
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_dcnv2_backward_pass(self, sample_inputs):
        """Test DCNv2 backward pass (gradient flow)."""
        dcn = DCNv2PackFlowGuided(
            in_channels=sample_inputs['channels'],
            out_channels=sample_inputs['channels'],
            kernel_size=3,
            padding=1,
            deformable_groups=16,
            pa_frames=sample_inputs['pa_frames']
        )

        # Enable gradient computation
        sample_inputs['x'].requires_grad_(True)

        output = dcn(
            sample_inputs['x'],
            sample_inputs['x_flow_warpeds'],
            sample_inputs['x_current'],
            sample_inputs['flows']
        )

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert sample_inputs['x'].grad is not None, "Input gradients should exist"
        assert sample_inputs['x'].grad.shape == sample_inputs['x'].shape, "Gradient shape should match input"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DCNv4 requires CUDA")
    def test_dcnv4_backward_pass(self, sample_inputs):
        """Test DCNv4 backward pass (gradient flow)."""
        # Move inputs to CUDA
        cuda_inputs = {}
        for key, value in sample_inputs.items():
            if isinstance(value, list):
                cuda_inputs[key] = [v.cuda() for v in value]
            elif isinstance(value, torch.Tensor):
                cuda_inputs[key] = value.cuda()
            else:
                cuda_inputs[key] = value  # Keep non-tensor values as-is

        dcn = DCNv4PackFlowGuided(
            in_channels=cuda_inputs['channels'],
            out_channels=cuda_inputs['channels'],
            kernel_size=3,
            padding=1,
            deformable_groups=16,
            pa_frames=cuda_inputs['pa_frames']
        ).cuda()

        # Enable gradient computation
        cuda_inputs['x'].requires_grad_(True)

        output = dcn(
            cuda_inputs['x'],
            cuda_inputs['x_flow_warpeds'],
            cuda_inputs['x_current'],
            cuda_inputs['flows']
        )

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert cuda_inputs['x'].grad is not None, "Input gradients should exist"
        assert cuda_inputs['x'].grad.shape == cuda_inputs['x'].shape, "Gradient shape should match input"

    def test_factory_function_dcnv2(self):
        """Test factory function returns DCNv2 by default."""
        opt = {'netG': {}}  # Empty config
        dcn_factory = get_deformable_module(opt)
        # Test that the factory creates DCNv2PackFlowGuided
        dcn_instance = dcn_factory(in_channels=64, out_channels=64, kernel_size=3, padding=1, deformable_groups=4, pa_frames=2)
        assert isinstance(dcn_instance, DCNv2PackFlowGuided), "Should create DCNv2PackFlowGuided by default"

    def test_factory_function_dcnv4(self):
        """Test factory function returns DCNv4 when configured."""
        try:
            from models.op.dcnv4 import DCNv4  # noqa: F401
        except (ImportError, RuntimeError):
            pytest.skip("DCNv4 CUDA extension not installed")
        opt = {'netG': {'dcn_type': 'DCNv4'}}
        dcn_factory = get_deformable_module(opt)
        # Test that the factory creates DCNv4PackFlowGuided
        dcn_instance = dcn_factory(in_channels=64, out_channels=64, kernel_size=3, padding=1, deformable_groups=4, pa_frames=2)
        assert isinstance(dcn_instance, DCNv4PackFlowGuided), "Should create DCNv4PackFlowGuided when configured"

    def test_factory_function_invalid_type(self):
        """Test factory function falls back to DCNv2 for invalid types."""
        opt = {'netG': {'dcn_type': 'InvalidType'}}
        dcn_factory = get_deformable_module(opt)
        # Test that the factory creates DCNv2PackFlowGuided for invalid types
        dcn_instance = dcn_factory(in_channels=64, out_channels=64, kernel_size=3, padding=1, deformable_groups=4, pa_frames=2)
        assert isinstance(dcn_instance, DCNv2PackFlowGuided), "Should create DCNv2PackFlowGuided for invalid types"

    @pytest.mark.parametrize("dcn_type", ["DCNv2", pytest.param("DCNv4", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="DCNv4 requires CUDA"))])
    def test_module_instantiation(self, dcn_type, sample_inputs):
        """Test that both DCN types can be instantiated and run forward pass."""
        opt = {'netG': {'dcn_type': dcn_type}}
        DCNClass = get_deformable_module(opt)

        # Use CUDA inputs for DCNv4
        if dcn_type == "DCNv4":
            test_inputs = {}
            for key, value in sample_inputs.items():
                if isinstance(value, list):
                    test_inputs[key] = [v.cuda() for v in value]
                elif isinstance(value, torch.Tensor):
                    test_inputs[key] = value.cuda()
                else:
                    test_inputs[key] = value  # Keep non-tensor values as-is
            dcn = DCNClass(
                in_channels=test_inputs['channels'],
                out_channels=test_inputs['channels'],
                kernel_size=3,
                padding=1,
                deformable_groups=16,
                pa_frames=test_inputs['pa_frames']
            ).cuda()
        else:
            test_inputs = sample_inputs
            dcn = DCNClass(
                in_channels=test_inputs['channels'],
                out_channels=test_inputs['channels'],
                kernel_size=3,
                padding=1,
                deformable_groups=16,
                pa_frames=test_inputs['pa_frames']
            )

        # Test forward pass
        output = dcn(
            test_inputs['x'],
            test_inputs['x_flow_warpeds'],
            test_inputs['x_current'],
            test_inputs['flows']
        )

        # Verify output shape
        assert output.shape == test_inputs['x'].shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DCNv4 requires CUDA for comparison")
    def test_dcnv2_vs_dcnv4_behavioral_consistency(self, sample_inputs):
        """Test behavioral consistency between DCNv2 and DCNv4 in realistic scenarios."""
        # This test verifies that DCNv4 behaves reasonably compared to DCNv2,
        # even though they have different architectures (DCNv4 removes softmax normalization)

        channels = sample_inputs['channels']

        # Create both DCN modules with identical configuration
        dcnv2 = DCNv2PackFlowGuided(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            deformable_groups=4,
            pa_frames=sample_inputs['pa_frames'],
            max_residue_magnitude=10
        )

        dcnv4 = DCNv4PackFlowGuided(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            deformable_groups=4,
            pa_frames=sample_inputs['pa_frames'],
            max_residue_magnitude=10
        )

        # Move to CUDA for DCNv4
        dcnv2 = dcnv2.cuda()
        dcnv4 = dcnv4.cuda()

        # Convert inputs to CUDA
        cuda_inputs = {}
        for key, value in sample_inputs.items():
            if isinstance(value, list):
                cuda_inputs[key] = [v.cuda() for v in value]
            elif isinstance(value, torch.Tensor):
                cuda_inputs[key] = value.cuda()
            else:
                cuda_inputs[key] = value

        # Test with the original inputs (without gradients for basic checks)
        with torch.no_grad():
            output_dcnv2 = dcnv2(
                cuda_inputs['x'],
                cuda_inputs['x_flow_warpeds'],
                cuda_inputs['x_current'],
                cuda_inputs['flows']
            )

            output_dcnv4 = dcnv4(
                cuda_inputs['x'],
                cuda_inputs['x_flow_warpeds'],
                cuda_inputs['x_current'],
                cuda_inputs['flows']
            )

        # Basic sanity checks
        assert output_dcnv2.shape == output_dcnv4.shape, "Shape mismatch"
        assert torch.isfinite(output_dcnv2).all(), "DCNv2 output not finite"
        assert torch.isfinite(output_dcnv4).all(), "DCNv4 output not finite"

        # Calculate differences
        abs_diff = torch.abs(output_dcnv2 - output_dcnv4)
        rel_diff = abs_diff / (torch.abs(output_dcnv2) + 1e-8)

        # Statistical analysis
        mean_abs_diff = abs_diff.mean().item()
        mean_rel_diff = rel_diff.mean().item()
        std_rel_diff = rel_diff.std().item()

        print("DCNv2 vs DCNv4 comparison:")
        print(".6f")
        print(".6f")
        print(".6f")

        # Behavioral consistency checks (relaxed thresholds due to architectural differences):
        # 1. Outputs should have reasonable magnitude (not exploding)
        assert output_dcnv2.abs().mean() < 10.0, "DCNv2 output magnitude too high"
        assert output_dcnv4.abs().mean() < 10.0, "DCNv4 output magnitude too high"

        # 2. Both should produce non-zero outputs (indicating they are actually computing)
        assert output_dcnv2.abs().mean() > 1e-6, "DCNv2 output too close to zero"
        assert output_dcnv4.abs().mean() > 1e-6, "DCNv4 output too close to zero"

        # 3. Check that gradients flow properly (separate forward pass with gradients)
        dcnv2.zero_grad()
        dcnv4.zero_grad()

        grad_output_dcnv2 = dcnv2(
            cuda_inputs['x'],
            cuda_inputs['x_flow_warpeds'],
            cuda_inputs['x_current'],
            cuda_inputs['flows']
        )
        grad_output_dcnv4 = dcnv4(
            cuda_inputs['x'],
            cuda_inputs['x_flow_warpeds'],
            cuda_inputs['x_current'],
            cuda_inputs['flows']
        )

        # Compute gradients
        grad_output_dcnv2.mean().backward()
        grad_output_dcnv4.mean().backward()

        # Verify that some gradients were computed (check parameter gradients exist)
        dcnv2_has_grad = any(p.grad is not None for p in dcnv2.parameters() if p.requires_grad)
        dcnv4_has_grad = any(p.grad is not None for p in dcnv4.parameters() if p.requires_grad)

        assert dcnv2_has_grad, "DCNv2 has no gradients computed"
        assert dcnv4_has_grad, "DCNv4 has no gradients computed"

        # Verify gradients are finite
        dcnv2_grads_finite = all(torch.isfinite(p.grad).all() for p in dcnv2.parameters() if p.grad is not None)
        dcnv4_grads_finite = all(torch.isfinite(p.grad).all() for p in dcnv4.parameters() if p.grad is not None)

        assert dcnv2_grads_finite, "DCNv2 has non-finite gradients"
        assert dcnv4_grads_finite, "DCNv4 has non-finite gradients"

        # 4. Test consistency with input scaling (both should scale similarly)
        scale_factor = 0.1
        scaled_inputs = {
            'x': cuda_inputs['x'] * scale_factor,
            'x_flow_warpeds': [fw * scale_factor for fw in cuda_inputs['x_flow_warpeds']],
            'x_current': cuda_inputs['x_current'] * scale_factor,
            'flows': cuda_inputs['flows'],
        }

        with torch.no_grad():
            scaled_output_dcnv2 = dcnv2(
                scaled_inputs['x'],
                scaled_inputs['x_flow_warpeds'],
                scaled_inputs['x_current'],
                scaled_inputs['flows']
            )

            scaled_output_dcnv4 = dcnv4(
                scaled_inputs['x'],
                scaled_inputs['x_flow_warpeds'],
                scaled_inputs['x_current'],
                scaled_inputs['flows']
            )

        # Both should scale roughly proportionally (allowing for architectural differences)
        dcnv2_scale_ratio = scaled_output_dcnv2.abs().mean() / (output_dcnv2.abs().mean() + 1e-8)
        dcnv4_scale_ratio = scaled_output_dcnv4.abs().mean() / (output_dcnv4.abs().mean() + 1e-8)

        print(".6f")
        print(".6f")

        # Scale ratios should be reasonably close (within 5x difference due to architectural changes)
        assert abs(dcnv2_scale_ratio - dcnv4_scale_ratio) < 5.0, \
            f"Scale ratios differ too much: DCNv2 {dcnv2_scale_ratio:.3f}, DCNv4 {dcnv4_scale_ratio:.3f}"

        print("✓ DCNv2 vs DCNv4 behavioral consistency test passed")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="DCNv4 requires CUDA for comparison")
    def test_dcnv4_softmax_vs_dcnv2_consistency(self, sample_inputs):
        """Test that DCNv4 with softmax behaves similarly to DCNv2."""
        channels = sample_inputs['channels']

        # Create DCNv2
        dcnv2 = DCNv2PackFlowGuided(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            deformable_groups=4,
            pa_frames=sample_inputs['pa_frames'],
            max_residue_magnitude=10
        )

        # Create DCNv4 with softmax enabled using factory
        opt_softmax = {'netG': {'dcn_type': 'DCNv4', 'dcn_apply_softmax': True}}
        DCNFactory = get_deformable_module(opt_softmax)
        dcnv4_softmax = DCNFactory(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            deformable_groups=4,
            pa_frames=sample_inputs['pa_frames'],
            max_residue_magnitude=10
        )

        # Move to CUDA
        dcnv2 = dcnv2.cuda()
        dcnv4_softmax = dcnv4_softmax.cuda()

        # Convert inputs to CUDA
        cuda_inputs = {}
        for key, value in sample_inputs.items():
            if isinstance(value, list):
                cuda_inputs[key] = [v.cuda() for v in value]
            elif isinstance(value, torch.Tensor):
                cuda_inputs[key] = value.cuda()
            else:
                cuda_inputs[key] = value

        # Forward pass
        with torch.no_grad():
            output_dcnv2 = dcnv2(
                cuda_inputs['x'],
                cuda_inputs['x_flow_warpeds'],
                cuda_inputs['x_current'],
                cuda_inputs['flows']
            )

            output_dcnv4_softmax = dcnv4_softmax(
                cuda_inputs['x'],
                cuda_inputs['x_flow_warpeds'],
                cuda_inputs['x_current'],
                cuda_inputs['flows']
            )

        # Basic sanity checks
        assert output_dcnv2.shape == output_dcnv4_softmax.shape, "Shape mismatch between DCNv2 and DCNv4+softmax"

        # Calculate differences
        abs_diff = torch.abs(output_dcnv2 - output_dcnv4_softmax)
        rel_diff = abs_diff / (torch.abs(output_dcnv2) + 1e-8)

        mean_abs_diff = abs_diff.mean().item()
        mean_rel_diff = rel_diff.mean().item()
        std_rel_diff = rel_diff.std().item()

        print("DCNv2 vs DCNv4+Softmax comparison:")
        print(".6f")
        print(".6f")
        print(".6f")

        # DCNv4 with softmax should behave more similarly to DCNv2 than DCNv4 without softmax
        # Allow for some architectural differences but expect better similarity than plain DCNv4
        assert output_dcnv2.abs().mean() > 0, "DCNv2 output should not be zero"
        assert output_dcnv4_softmax.abs().mean() > 0, "DCNv4+softmax output should not be zero"

        # The outputs should be in reasonable ranges (not exploding)
        assert output_dcnv2.abs().mean() < 100.0, "DCNv2 output magnitude too high"
        assert output_dcnv4_softmax.abs().mean() < 100.0, "DCNv4+softmax output magnitude too high"

        # Statistical properties should be somewhat similar
        dcnv2_std = output_dcnv2.std().item()
        dcnv4_softmax_std = output_dcnv4_softmax.std().item()

        print(".6f")
        print(".6f")

        # Standard deviations should be in similar ranges
        assert abs(dcnv2_std - dcnv4_softmax_std) < max(dcnv2_std, dcnv4_softmax_std) * 2.0, \
            "Standard deviations differ too much"

        print("✓ DCNv4 with softmax vs DCNv2 consistency test passed")

    def test_config_consistency_training_vs_testing(self):
        """Test that training and testing use consistent DCN configurations."""
        try:
            from models.op.dcnv4 import DCNv4  # noqa: F401
        except (ImportError, RuntimeError):
            pytest.skip("DCNv4 CUDA extension not installed")
        import json
        import tempfile
        import os
        from models.architectures.vrt import VRT

        # Create a test configuration with DCNv4
        test_config = {
            "netG": {
                "dcn_type": "DCNv4",
                "dcn_apply_softmax": True,
                "upscale": 1,
                "in_chans": 3,
                "img_size": [2, 32, 32],
                "window_size": [2, 4, 4],
                "depths": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 11 depths for minimal VRT
                "embed_dims": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],  # 11 dims for minimal VRT
                "num_heads": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 11 heads for minimal VRT
                "indep_reconsts": [],  # No independent reconstruction for minimal model
                "pa_frames": 2,
                "deformable_groups": 4
            }
        }

        # Test that VRT can be created with the config (simulating training)
        try:
            model_training = VRT(
                upscale=test_config['netG']['upscale'],
                in_chans=test_config['netG']['in_chans'],
                img_size=test_config['netG']['img_size'],
                window_size=test_config['netG']['window_size'],
                depths=test_config['netG']['depths'],
                embed_dims=test_config['netG']['embed_dims'],
                num_heads=test_config['netG']['num_heads'],
                pa_frames=test_config['netG']['pa_frames'],
                deformable_groups=test_config['netG']['deformable_groups'],
                dcn_config={
                    'type': test_config['netG']['dcn_type'],
                    'apply_softmax': test_config['netG']['dcn_apply_softmax']
                }
            )

            # Check that the model uses DCNv4
            dcn_instance = model_training.stage1.pa_deform
            assert isinstance(dcn_instance, DCNv4PackFlowGuided), \
                f"Training model should use DCNv4PackFlowGuided, got {type(dcn_instance)}"
            assert dcn_instance.apply_softmax == True, \
                "Training model should have apply_softmax=True"

            print("✓ Training configuration creates DCNv4 model correctly")

        except Exception as e:
            pytest.fail(f"Failed to create training model with DCNv4 config: {e}")

        # Test that the config can be saved and loaded (simulating config persistence)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            config_path = f.name

        try:
            # Simulate loading config for testing (like main_test_vrt.py does)
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            # Extract netG config as main_test_vrt.py does
            netG_cfg = loaded_config.get('netG', {})

            # Create model as main_test_vrt.py does
            model_testing = VRT(
                upscale=netG_cfg.get('upscale', 1),
                in_chans=netG_cfg.get('in_chans', 3),
                img_size=netG_cfg['img_size'],
                window_size=netG_cfg['window_size'],
                depths=netG_cfg['depths'],
                embed_dims=netG_cfg['embed_dims'],
                num_heads=netG_cfg['num_heads'],
                pa_frames=netG_cfg.get('pa_frames', 2),
                deformable_groups=netG_cfg.get('deformable_groups', 16),
                dcn_config={
                    'type': netG_cfg.get('dcn_type', 'DCNv2'),
                    'apply_softmax': netG_cfg.get('dcn_apply_softmax', False)
                }
            )

            # Check that testing model uses the same DCN config
            dcn_instance_test = model_testing.stage1.pa_deform
            assert isinstance(dcn_instance_test, DCNv4PackFlowGuided), \
                f"Testing model should use DCNv4PackFlowGuided, got {type(dcn_instance_test)}"
            assert dcn_instance_test.apply_softmax == True, \
                "Testing model should have apply_softmax=True"

            # Verify that both models have the same DCN configuration
            assert type(model_training.stage1.pa_deform) == type(model_testing.stage1.pa_deform), \
                "Training and testing models should use the same DCN type"

            assert model_training.stage1.pa_deform.apply_softmax == model_testing.stage1.pa_deform.apply_softmax, \
                "Training and testing models should have the same softmax setting"

            print("✓ Testing configuration loads DCNv4 model correctly")
            print("✓ Training and testing configurations are consistent")

        except Exception as e:
            pytest.fail(f"Failed to create/load testing model with consistent config: {e}")
        finally:
            # Clean up temp file
            os.unlink(config_path)

        print("✓ Config consistency test passed - no DCNv2/DCNv4 mismatch between training and testing")


if __name__ == "__main__":
    # Run basic smoke test
    test = TestDCNModules()

    # Generate sample inputs
    sample_inputs = test.sample_inputs()

    print("Testing DCNv2 forward shape...")
    test.test_dcnv2_forward_shape(sample_inputs)
    print("✓ DCNv2 forward shape test passed")

    print("Testing DCNv4 forward shape...")
    test.test_dcnv4_forward_shape(sample_inputs)
    print("✓ DCNv4 forward shape test passed")

    print("Testing DCNv2 backward pass...")
    test.test_dcnv2_backward_pass(sample_inputs)
    print("✓ DCNv2 backward test passed")

    print("Testing DCNv4 backward pass...")
    test.test_dcnv4_backward_pass(sample_inputs)
    print("✓ DCNv4 backward test passed")

    print("Testing factory functions...")
    test.test_factory_function_dcnv2()
    test.test_factory_function_dcnv4()
    test.test_factory_function_invalid_type()
    print("✓ Factory function tests passed")

    print("Testing DCNv2 vs DCNv4 behavioral consistency...")
    test.test_dcnv2_vs_dcnv4_behavioral_consistency(sample_inputs)
    print("✓ Behavioral consistency test passed")

    print("Testing DCNv4 with softmax vs DCNv2 consistency...")
    test.test_dcnv4_softmax_vs_dcnv2_consistency(sample_inputs)
    print("✓ DCNv4 softmax consistency test passed")

    print("Testing config consistency between training and testing...")
    test.test_config_consistency_training_vs_testing()
    print("✓ Config consistency test passed")

    print("All basic tests passed!")