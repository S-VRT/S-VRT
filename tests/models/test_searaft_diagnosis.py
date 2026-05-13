"""
SeaRAFT 诊断测试套件

用于诊断 SeaRAFT 与 SpyNet 之间的不一致性：
1. 输入范围 sanity check
2. 平移合成测试
3. 多尺度幅值测试
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from models.optical_flow import create_optical_flow
from models.optical_flow.base import OpticalFlowModule
from models.utils.flow import flow_warp


class TestSeaRaftDiagnosis:
    """SeaRAFT 诊断测试类"""

    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @pytest.fixture
    def sample_frames(self, device):
        """创建测试帧 (BGR [0,1] 格式, VRT 标准输入)"""
        b, c, h, w = 1, 3, 128, 128
        # 创建有内容的测试图像，而不是纯噪声
        x = torch.linspace(0, 1, w, device=device).unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
        y = torch.linspace(0, 1, h, device=device).unsqueeze(0).unsqueeze(-1).expand(b, 1, h, w)
        frame = (x + y) / 2  # 对角线渐变 [B, 1, H, W]
        frame = frame.expand(b, c, h, w)  # 扩展到 [B, C, H, W]
        frame1 = frame.clone()
        frame2 = frame.clone() + 0.05  # 轻微变化
        return frame1, frame2

    def test_input_range_sanity_check(self, sample_frames, device):
        """测试 1: 输入范围 sanity check - 检查送入 SeaRAFT 的张量 min/max 和 /255 预处理"""

        frame1, frame2 = sample_frames

        print("\n=== 输入范围 Sanity Check ===")
        print(f"原始输入 (VRT 格式): BGR [0,1]")
        print(f"frame1 范围: [{frame1.min().item():.6f}, {frame1.max().item():.6f}]")
        print(f"frame2 范围: [{frame2.min().item():.6f}, {frame2.max().item():.6f}]")

        # 手动测试预处理步骤 (因为 preprocess_frames 是静态方法)
        print("\n--- 测试 SeaRAFT 预处理 ---")
        from models.optical_flow.base import OpticalFlowModule

        # SeaRAFT 使用 'rgb_255' 格式
        preprocessed1, preprocessed2 = OpticalFlowModule.preprocess_frames(frame1, frame2, 'rgb_255')
        print(f"SeaRAFT 预处理后 (rgb_255):")
        print(f"  frame1 范围: [{preprocessed1.min().item():.6f}, {preprocessed1.max().item():.6f}]")
        print(f"  frame2 范围: [{preprocessed2.min().item():.6f}, {preprocessed2.max().item():.6f}]")

        max_val = preprocessed1.max().item()
        if max_val > 1.0:
            print(f"  ✓ 已应用 /255 预处理 (最大值 {max_val:.3f} > 1.0)")
        else:
            print(f"  ✗ 可能未正确应用 /255 (最大值 {max_val:.3f})")

        # 对比 SpyNet 预处理
        print("\n--- 测试 SpyNet 预处理 ---")
        spynet_prep1, spynet_prep2 = OpticalFlowModule.preprocess_frames(frame1, frame2, 'rgb_norm')
        print(f"SpyNet 预处理后 (rgb_norm):")
        print(f"  frame1 范围: [{spynet_prep1.min().item():.6f}, {spynet_prep1.max().item():.6f}]")
        print(f"  frame2 范围: [{spynet_prep2.min().item():.6f}, {spynet_prep2.max().item():.6f}]")
        print("  ✓ 使用 ImageNet 归一化")

    def test_translation_synthesis(self, device):
        """测试 2: 平移合成测试 - 比较 SpyNet vs SeaRAFT 对已知平移的响应"""

        print("\n=== 平移合成测试 ===")

        b, c, h, w = 1, 3, 128, 128
        tx, ty = 5.0, -3.0  # 已知平移: 右移 5px, 上移 3px (ty 负值表示向上)

        # 创建基准图像 (有清晰的模式以便光流检测)
        x = torch.linspace(0, 1, w, device=device).unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
        y = torch.linspace(0, 1, h, device=device).unsqueeze(0).unsqueeze(-1).expand(b, 1, h, w)
        pattern = torch.sin(x * 10) * torch.cos(y * 10)  # 创建有规律的模式 [B, 1, H, W]
        base_image = pattern.expand(b, c, h, w)  # 扩展到 [B, C, H, W]

        # 使用 F.grid_sample 创建精确的平移
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # 创建反向平移网格 (因为我们想要从 base_image 到 translated_image 的变换)
        # 如果 translated_image 是 base_image 右移 tx, 则 flow 应该是 (-tx, -ty)
        translate_grid = grid.clone()
        translate_grid[..., 0] -= tx  # 反向 x 平移
        translate_grid[..., 1] -= ty  # 反向 y 平移

        # 归一化到 [-1, 1]
        translate_grid[..., 0] = 2.0 * translate_grid[..., 0] / (w - 1) - 1.0
        translate_grid[..., 1] = 2.0 * translate_grid[..., 1] / (h - 1) - 1.0

        translated_image = F.grid_sample(base_image, translate_grid,
                                       mode='bilinear', padding_mode='zeros', align_corners=True)

        print(f"合成平移: tx={tx}, ty={ty} (正值表示右/下方向)")
        print(f"基准图像范围: [{base_image.min().item():.6f}, {base_image.max().item():.6f}]")
        print(f"平移图像范围: [{translated_image.min().item():.6f}, {translated_image.max().item():.6f}]")

        # 预期 flow: 从 base_image 到 translated_image, flow 应该是 (tx, ty)
        # 因为 translated_image 是 base_image 向右/下平移 tx/ty 得到的
        expected_dx, expected_dy = tx, ty

        results = {}

        # 测试 SpyNet
        try:
            spynet = create_optical_flow('spynet', device=device)
            spynet_flows = spynet(base_image, translated_image)
            spynet_flow = spynet_flows[0] if isinstance(spynet_flows, list) else spynet_flows

            print("\nSpyNet 结果:")
            print(f"  输出形状: {spynet_flow.shape}")
            print(f"  dx 范围: [{spynet_flow[:, 0].min().item():.6f}, {spynet_flow[:, 0].max().item():.6f}]")
            print(f"  dy 范围: [{spynet_flow[:, 1].min().item():.6f}, {spynet_flow[:, 1].max().item():.6f}]")

            dx_actual = spynet_flow[:, 0].mean().item()
            dy_actual = spynet_flow[:, 1].mean().item()
            print(".6f")

            dx_error = abs(dx_actual - expected_dx)
            dy_error = abs(dy_actual - expected_dy)
            print(".6f")

            results['spynet'] = {
                'dx_actual': dx_actual, 'dy_actual': dy_actual,
                'dx_error': dx_error, 'dy_error': dy_error
            }

        except Exception as e:
            print(f"SpyNet 测试失败: {e}")
            results['spynet'] = None

        # 测试 SeaRAFT
        try:
            sea_raft = create_optical_flow('sea_raft', device=device)
            raft_flows = sea_raft(base_image, translated_image)
            raft_flow = raft_flows[0] if isinstance(raft_flows, list) else raft_flows

            print("\nSeaRAFT 结果:")
            print(f"  输出形状: {raft_flow.shape}")
            print(f"  dx 范围: [{raft_flow[:, 0].min().item():.6f}, {raft_flow[:, 0].max().item():.6f}]")
            print(f"  dy 范围: [{raft_flow[:, 1].min().item():.6f}, {raft_flow[:, 1].max().item():.6f}]")

            dx_actual = raft_flow[:, 0].mean().item()
            dy_actual = raft_flow[:, 1].mean().item()
            print(".6f")

            dx_error = abs(dx_actual - expected_dx)
            dy_error = abs(dy_actual - expected_dy)
            print(".6f")

            results['sea_raft'] = {
                'dx_actual': dx_actual, 'dy_actual': dy_actual,
                'dx_error': dx_error, 'dy_error': dy_error
            }

        except Exception as e:
            print(f"SeaRAFT 测试失败: {e}")
            results['sea_raft'] = None

        # 比较结果
        if results['spynet'] and results['sea_raft']:
            print("\n=== 比较结果 ===")
            spy_dx_err = results['spynet']['dx_error']
            raft_dx_err = results['sea_raft']['dx_error']
            spy_dy_err = results['spynet']['dy_error']
            raft_dy_err = results['sea_raft']['dy_error']

            print(".6f")
            print(".6f")

            if raft_dx_err > spy_dx_err * 2 or raft_dy_err > spy_dy_err * 2:
                print("⚠️  SeaRAFT 的平移估计误差显著大于 SpyNet!")
            else:
                print("✓ SeaRAFT 和 SpyNet 的平移估计误差相近")

    def test_multiscale_flow_amplitude(self, sample_frames, device):
        """测试 3: 多尺度幅值测试 - 测试 flow 下采样后的 warp 误差和缩放校正"""

        print("\n=== 多尺度幅值测试 ===")

        frame1, frame2 = sample_frames
        b, c, h, w = frame1.shape

        # 创建测试特征图 (模拟 VRT 中的特征)
        test_feature = torch.randn(b, 64, h, w, device=device)  # 模拟特征图

        # 获取 SpyNet 的多尺度输出
        try:
            spynet = create_optical_flow('spynet', device=device)
            spynet_flows = spynet(frame1, frame2)

            if isinstance(spynet_flows, list):
                print(f"SpyNet 返回 {len(spynet_flows)} 个尺度")

                # 使用最精细的尺度作为基准
                full_res_flow = spynet_flows[0]  # 假设第一个是最精细的
                print(f"全分辨率 flow 形状: {full_res_flow.shape}")

                # 基准: 用全分辨率 flow warp
                warped_full = flow_warp(test_feature, full_res_flow.permute(0, 2, 3, 1), 'bilinear')
                target = test_feature  # 理想情况下应该是相同的

                self._test_multiscale_flow_impl(full_res_flow, test_feature, target, "SpyNet", h, w)

            else:
                print("SpyNet 没有返回多尺度输出")

        except Exception as e:
            print(f"SpyNet 多尺度测试失败: {e}")

        # 测试 SeaRAFT
        try:
            sea_raft = create_optical_flow('sea_raft', device=device)
            raft_flows = sea_raft(frame1, frame2)

            if isinstance(raft_flows, list):
                print(f"\nSeaRAFT 返回 {len(raft_flows)} 个尺度")

                full_res_flow = raft_flows[0]
                print(f"全分辨率 flow 形状: {full_res_flow.shape}")

                # 基准: 用全分辨率 flow warp
                warped_full = flow_warp(test_feature, full_res_flow.permute(0, 2, 3, 1), 'bilinear')
                target = test_feature  # 理想情况下应该是相同的

                self._test_multiscale_flow_impl(full_res_flow, test_feature, target, "SeaRAFT", h, w)

            else:
                print("SeaRAFT 没有返回多尺度输出")

        except Exception as e:
            print(f"SeaRAFT 多尺度测试失败: {e}")

    def _test_multiscale_flow_impl(self, full_res_flow, test_feature, target, model_name, h, w):
        """多尺度 flow 测试的辅助函数"""
        for scale_factor in [2, 4]:
            print(f"\n--- {model_name} 尺度 1/{scale_factor} 测试 ---")

            # 下采样 flow (不校正幅值)
            h_ds = h // scale_factor
            w_ds = w // scale_factor
            flow_downsampled = F.interpolate(full_res_flow, size=(h_ds, w_ds), mode='bilinear', align_corners=False)

            # 下采样特征图到对应尺度
            feature_ds = F.interpolate(test_feature, size=(h_ds, w_ds), mode='bilinear', align_corners=False)

            # 用下采样 flow warp 下采样特征
            warped_ds = flow_warp(feature_ds, flow_downsampled.permute(0, 2, 3, 1), 'bilinear')

            # 计算误差 (与下采样 target 比较)
            target_ds = F.interpolate(target, size=(h_ds, w_ds), mode='bilinear', align_corners=False)
            error_no_correction = F.mse_loss(warped_ds, target_ds).item()
            print(".8f")

            # 尝试幅值校正: flow / scale_factor
            flow_corrected = flow_downsampled / scale_factor

            warped_corrected = flow_warp(feature_ds, flow_corrected.permute(0, 2, 3, 1), 'bilinear')
            error_with_correction = F.mse_loss(warped_corrected, target_ds).item()
            print(".8f")

            if error_no_correction > 1e-10:  # 避免除零
                improvement = (error_no_correction - error_with_correction) / error_no_correction * 100
                print(".2f")
            else:
                print("  误差太小，无法计算改进百分比")


def test_searaft_vrt_compatibility_summary():
    """总结 SeaRAFT 与 VRT 兼容性问题的测试"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*60)
    print("SeaRAFT vs SpyNet 兼容性诊断总结")
    print("="*60)

    # 测试输入预处理差异
    frame1 = torch.rand(1, 3, 64, 64, device=device)
    frame2 = torch.rand(1, 3, 64, 64, device=device)

    print("\n1. 输入预处理差异:")
    raft_prep1, raft_prep2 = OpticalFlowModule.preprocess_frames(frame1, frame2, 'rgb_255')
    spy_prep1, spy_prep2 = OpticalFlowModule.preprocess_frames(frame1, frame2, 'rgb_norm')

    print(".3f")
    print(".3f")

    # 测试平移估计准确性
    print("\n2. 平移估计准确性:")
    # 创建已知平移的测试数据
    b, c, h, w = 1, 3, 64, 64
    tx, ty = 3.0, 2.0

    x = torch.linspace(0, 1, w, device=device).unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    y = torch.linspace(0, 1, h, device=device).unsqueeze(0).unsqueeze(-1).expand(b, 1, h, w)
    pattern = torch.sin(x * 5) * torch.cos(y * 5)
    base_image = pattern.expand(b, c, h, w)

    # 创建平移后的图像
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    translate_grid = grid.clone()
    translate_grid[..., 0] -= tx
    translate_grid[..., 1] -= ty
    translate_grid[..., 0] = 2.0 * translate_grid[..., 0] / (w - 1) - 1.0
    translate_grid[..., 1] = 2.0 * translate_grid[..., 1] / (h - 1) - 1.0
    translated_image = F.grid_sample(base_image, translate_grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    # 测试两个模型
    spynet = create_optical_flow('spynet', device=device)
    sea_raft = create_optical_flow('sea_raft', device=device)

    spy_flow = spynet(base_image, translated_image)[0]
    raft_flow = sea_raft(base_image, translated_image)[0]

    spy_dx = spy_flow[:, 0].mean().item()
    spy_dy = spy_flow[:, 1].mean().item()
    raft_dx = raft_flow[:, 0].mean().item()
    raft_dy = raft_flow[:, 1].mean().item()

    print(".3f")
    print(".3f")
    print(".3f")

    spy_error = ((spy_dx - tx)**2 + (spy_dy - ty)**2)**0.5
    raft_error = ((raft_dx - tx)**2 + (raft_dy - ty)**2)**0.5

    print(".3f")
    print(".3f")

    # 诊断结论
    print("\n3. 诊断结论:")
    issues = []

    if raft_prep1.max() > 10:  # SeaRAFT 使用 [0,255]
        issues.append("✓ 输入预处理: SeaRAFT 使用 [0,255] 而非 ImageNet 归一化")

    if raft_error > spy_error * 5:
        issues.append("✓ 平移估计: SeaRAFT 方向/幅值与预期不符")

    # 测试多尺度校正效果
    test_feature = torch.randn(1, 32, h, w, device=device)
    flow_ds = F.interpolate(raft_flow, size=(h//2, w//2), mode='bilinear', align_corners=False)
    feature_ds = F.interpolate(test_feature, size=(h//2, w//2), mode='bilinear', align_corners=False)

    # 不校正
    warped_no_corr = flow_warp(feature_ds, flow_ds.permute(0, 2, 3, 1), 'bilinear')
    target_ds = F.interpolate(test_feature, size=(h//2, w//2), mode='bilinear', align_corners=False)
    error_no_corr = F.mse_loss(warped_no_corr, target_ds).item()

    # 校正
    flow_corrected = flow_ds / 2.0
    warped_corrected = flow_warp(feature_ds, flow_corrected.permute(0, 2, 3, 1), 'bilinear')
    error_corrected = F.mse_loss(warped_corrected, target_ds).item()

    improvement = (error_no_corr - error_corrected) / error_no_corr * 100
    if improvement > 50:
        issues.append(".1f")

    for issue in issues:
        print(f"  {issue}")

    print("\n4. 建议修复:")
    print("  • 统一输入预处理格式 (考虑改为 ImageNet 归一化)")
    print("  • 校正 SeaRAFT 的坐标系统/单位以匹配 VRT 期望")
    print("  • 在多尺度使用时应用 /scale_factor 校正")
    print("  • 考虑减少 SeaRAFT 的激进后处理 (max_displacement 裁剪)")

    return issues


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
