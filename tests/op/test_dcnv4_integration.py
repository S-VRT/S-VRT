#!/usr/bin/env python3
"""
DCNv4集成测试脚本
验证DCNv4可以正常创建和运行
"""

import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dcnv4_creation():
    """测试DCNv4模块创建"""
    print("Testing DCNv4 module creation...")

    try:
        from models.op.dcnv4 import DCNv4
        dcn = DCNv4(channels=64, kernel_size=3, stride=1, pad=1, group=4)
        print("✓ DCNv4 module created successfully")
        return True
    except Exception as e:
        print(f"✗ DCNv4 module creation failed: {e}")
        return False

def test_dcn_factory():
    """测试DCN工厂函数"""
    print("Testing DCN factory function...")

    try:
        from models.blocks.dcn import get_deformable_module

        # 测试DCNv2
        opt_v2 = {'netG': {'dcn_type': 'DCNv2'}}
        DCNv2Class = get_deformable_module(opt_v2)
        print(f"✓ DCNv2 factory returned: {DCNv2Class.__name__}")

        # 测试DCNv4
        opt_v4 = {'netG': {'dcn_type': 'DCNv4'}}
        DCNv4Class = get_deformable_module(opt_v4)
        print(f"✓ DCNv4 factory returned: {DCNv4Class.__name__}")

        # 测试默认值
        opt_default = {'netG': {}}
        DCNDefault = get_deformable_module(opt_default)
        print(f"✓ Default DCN factory returned: {DCNDefault.__name__}")

        return True
    except Exception as e:
        print(f"✗ DCN factory test failed: {e}")
        return False

def test_dcn_adapter_creation():
    """测试DCN适配器创建"""
    print("Testing DCN adapter creation...")

    try:
        from models.blocks.dcn import DCNv4PackFlowGuided

        # 创建适配器实例
        dcn_adapter = DCNv4PackFlowGuided(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            deformable_groups=4,
            pa_frames=2,
            max_residue_magnitude=10
        )
        print("✓ DCNv4PackFlowGuided adapter created successfully")
        return True
    except Exception as e:
        print(f"✗ DCN adapter creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dcn_adapter_forward():
    """测试DCN适配器前向传播"""
    print("Testing DCN adapter forward pass...")

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping DCN adapter forward test")
        return True

    try:
        from models.blocks.dcn import DCNv4PackFlowGuided

        # 创建适配器
        dcn_adapter = DCNv4PackFlowGuided(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
            deformable_groups=4,
            pa_frames=2,
            max_residue_magnitude=10
        ).cuda()

        # 创建测试数据
        batch_size, channels, height, width = 1, 64, 16, 16

        # 输入特征 [B, C, H, W]
        x = torch.randn(batch_size, channels, height, width).cuda()

        # 模拟x_flow_warpeds, x_current, flows参数
        x_flow_warpeds = [torch.randn(batch_size, channels, height, width).cuda()]
        x_current = torch.randn(batch_size, channels, height, width).cuda()
        flows = [torch.randn(batch_size, 2, height, width).cuda()]

        # 执行前向传播
        output = dcn_adapter(x, x_flow_warpeds, x_current, flows)

        print(f"✓ DCNv4 adapter forward pass successful, output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"✗ DCN adapter forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("=== DCNv4 Integration Test ===\n")

    tests = [
        test_dcnv4_creation,
        test_dcn_factory,
        test_dcn_adapter_creation,
        test_dcn_adapter_forward
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"=== Test Results: {passed}/{total} tests passed ===")

    if passed == total:
        print("🎉 All DCNv4 integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
