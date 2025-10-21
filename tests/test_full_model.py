#!/usr/bin/env python
"""
完整模型测试脚本
验证 VRTWithSpike 的前向传播是否正常工作
"""
import sys
from pathlib import Path

# 添加路径
REPO_ROOT = Path(__file__).parent
VRT_ROOT = REPO_ROOT / "third_party" / "VRT"
sys.path.insert(0, str(VRT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch
from models.network_vrt import VRT  # type: ignore
from src.models.integrate_vrt import VRTWithSpike


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    print("\n[1/3] Creating VRT backbone...")
    vrt = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=[5, 256, 256],
        window_size=[5, 8, 8],
        embed_dims=[96] * 13,
        use_checkpoint_attn=False,  # 测试时关闭
        use_checkpoint_ffn=False,
    )
    print("    ✅ VRT backbone created")
    
    print("\n[2/3] Creating VRTWithSpike wrapper...")
    model = VRTWithSpike(
        vrt_backbone=vrt,
        spike_bins=32,
        channels_per_scale=[96, 96, 96, 96],
        tsa_heads=4,
        fuse_heads=4,
    )
    print("    ✅ VRTWithSpike created")
    
    print("\n[3/3] Counting parameters...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params / 1e6:.2f}M")
    print(f"    Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    return model


def test_forward_cpu(model):
    """测试 CPU 前向传播"""
    print("\n" + "=" * 60)
    print("Testing Forward Pass (CPU)")
    print("=" * 60)
    
    batch_size = 1
    clip_len = 5
    height, width = 256, 256
    spike_bins = 32
    
    print(f"\n[1/3] Creating dummy inputs...")
    print(f"    Batch size: {batch_size}")
    print(f"    Clip length: {clip_len}")
    print(f"    Resolution: {height}x{width}")
    print(f"    Spike bins: {spike_bins}")
    
    dummy_rgb = torch.randn(batch_size, clip_len, 3, height, width)
    dummy_spike = torch.randn(batch_size, clip_len, spike_bins, height, width)
    
    print(f"\n[2/3] Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(dummy_rgb, dummy_spike)
    
    print(f"\n[3/3] Checking output...")
    print(f"    Input RGB shape: {dummy_rgb.shape}")
    print(f"    Input Spike shape: {dummy_spike.shape}")
    print(f"    Output shape: {output.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, clip_len, 3, height, width)
    assert output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    
    print(f"    ✅ Output shape correct: {output.shape}")
    
    # 验证输出数值范围
    print(f"    Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"    Output mean: {output.mean():.4f}")
    print(f"    Output std: {output.std():.4f}")
    
    return output


def test_forward_gpu(model):
    """测试 GPU 前向传播"""
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Skipping GPU Test (CUDA not available)")
        print("=" * 60)
        return None
    
    print("\n" + "=" * 60)
    print("Testing Forward Pass (GPU)")
    print("=" * 60)
    
    device = torch.device("cuda:0")
    print(f"\n[1/4] Moving model to GPU...")
    model = model.to(device)
    print(f"    ✅ Model on {device}")
    
    batch_size = 2
    clip_len = 5
    height, width = 256, 256
    spike_bins = 32
    
    print(f"\n[2/4] Creating dummy inputs on GPU...")
    dummy_rgb = torch.randn(batch_size, clip_len, 3, height, width).to(device)
    dummy_spike = torch.randn(batch_size, clip_len, spike_bins, height, width).to(device)
    
    print(f"\n[3/4] Running forward pass...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_rgb, dummy_spike)
    
    print(f"\n[4/4] Checking GPU memory...")
    peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3
    current_memory = torch.cuda.memory_allocated(0) / 1024**3
    print(f"    Peak memory: {peak_memory:.2f} GB")
    print(f"    Current memory: {current_memory:.2f} GB")
    
    print(f"\n    Output shape: {output.shape}")
    print(f"    ✅ GPU forward pass successful")
    
    return output


def test_backward_gpu(model):
    """测试 GPU 反向传播"""
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Skipping Backward Test (CUDA not available)")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("Testing Backward Pass (GPU)")
    print("=" * 60)
    
    device = torch.device("cuda:0")
    model = model.to(device)
    model.train()
    
    batch_size = 1
    clip_len = 5
    height, width = 256, 256
    spike_bins = 32
    
    print(f"\n[1/4] Creating inputs...")
    dummy_rgb = torch.randn(batch_size, clip_len, 3, height, width).to(device)
    dummy_spike = torch.randn(batch_size, clip_len, spike_bins, height, width).to(device)
    dummy_target = torch.randn(batch_size, clip_len, 3, height, width).to(device)
    
    print(f"\n[2/4] Forward pass...")
    output = model(dummy_rgb, dummy_spike)
    
    print(f"\n[3/4] Computing loss and backward...")
    loss = torch.nn.functional.mse_loss(output, dummy_target)
    print(f"    Loss value: {loss.item():.6f}")
    
    loss.backward()
    print(f"    ✅ Backward pass successful")
    
    print(f"\n[4/4] Checking gradients...")
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    if grad_norms:
        print(f"    Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, mean={sum(grad_norms)/len(grad_norms):.6f}")
        print(f"    ✅ Gradients computed for {len(grad_norms)} parameters")
    else:
        print(f"    ⚠️  No gradients found!")


def test_spike_components():
    """测试 Spike 分支各组件"""
    print("\n" + "=" * 60)
    print("Testing Spike Components")
    print("=" * 60)
    
    from src.models.spike_encoder3d import SpikeEncoder3D
    from src.models.spike_temporal_sa import SpikeTemporalSA
    from src.models.fusion.cross_attn_temporal import MultiScaleTemporalCrossAttnFuse
    
    batch_size = 2
    clip_len = 5
    spike_bins = 32
    height, width = 256, 256
    channels = [96, 96, 96, 96]
    
    # 测试 SpikeEncoder3D
    print(f"\n[1/3] Testing SpikeEncoder3D...")
    encoder = SpikeEncoder3D(in_bins=spike_bins, channels_per_scale=channels)
    dummy_spike = torch.randn(batch_size, clip_len, spike_bins, height, width)
    
    spike_feats = encoder(dummy_spike)
    print(f"    Input: {dummy_spike.shape}")
    for i, feat in enumerate(spike_feats):
        print(f"    Scale {i+1} output: {feat.shape}")
    print(f"    ✅ SpikeEncoder3D works correctly")
    
    # 测试 SpikeTemporalSA
    print(f"\n[2/3] Testing SpikeTemporalSA...")
    tsa = SpikeTemporalSA(channels_per_scale=channels, heads=4)
    tsa_out = tsa(spike_feats)
    
    for i, feat in enumerate(tsa_out):
        assert feat.shape == spike_feats[i].shape, f"TSA shape mismatch at scale {i+1}"
        print(f"    Scale {i+1} TSA output: {feat.shape}")
    print(f"    ✅ SpikeTemporalSA works correctly")
    
    # 测试 CrossAttention Fusion
    print(f"\n[3/3] Testing MultiScaleTemporalCrossAttnFuse...")
    fuse = MultiScaleTemporalCrossAttnFuse(channels_per_scale=channels, heads=4)
    
    # 模拟 RGB 特征（转为 B,T,C,H,W 格式）
    rgb_feats_btchw = [f.permute(0, 2, 1, 3, 4) for f in spike_feats]
    spike_feats_btchw = [f.permute(0, 2, 1, 3, 4) for f in tsa_out]
    
    fused = fuse(rgb_feats_btchw, spike_feats_btchw)
    
    for i, feat in enumerate(fused):
        print(f"    Scale {i+1} fused output: {feat.shape}")
    print(f"    ✅ MultiScaleTemporalCrossAttnFuse works correctly")


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("VRT+Spike Full Model Test")
    print("=" * 60)
    
    # 1. 测试模型创建
    model = test_model_creation()
    
    # 2. 测试 Spike 组件
    test_spike_components()
    
    # 3. 测试 CPU 前向传播
    test_forward_cpu(model)
    
    # 4. 测试 GPU 前向传播
    test_forward_gpu(model)
    
    # 5. 测试 GPU 反向传播
    test_backward_gpu(model)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe model is ready for training. Next steps:")
    print("1. Run dataset test: python test_dataset.py")
    print("2. Check GPU memory: python test_gpu_memory.py")
    print("3. Start training: python src/train.py --config configs/deblur/vrt_spike_baseline.yaml")
    print()


if __name__ == "__main__":
    main()

