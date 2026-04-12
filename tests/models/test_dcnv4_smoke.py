#!/usr/bin/env python3
"""
DCNv4集成Smoke测试
验证DCNv4与VRT模型的完整集成
"""

import pytest
import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.smoke
class TestDCNv4Integration:
    """DCNv4集成测试类"""

    @pytest.fixture
    def minimal_vrt_config(self):
        """最小VRT配置用于测试"""
        return {
            'model': 'vrt',
            'is_train': True,
            'gpu_ids': [0],
            'dist': False,
            'scale': 1,
            'n_channels': 3,
            'path': {
                'models': './test_models',
                'log': './test_logs'
            },
            'netG': {
                'net_type': 'vrt',
                'in_chans': 3,
                'upscale': 1,
                'img_size': [4, 64, 64],  # 小尺寸用于测试
                'window_size': [4, 8, 8],
                'depths': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 很小的深度
                'indep_reconsts': [9, 10],
                'embed_dims': [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],  # 小embed_dim
                'num_heads': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                'spynet_path': None,
                'optical_flow': {'module': 'spynet', 'checkpoint': None, 'params': {}},
                'pa_frames': 2,
                'deformable_groups': 4,
                'nonblind_denoising': False,
                'use_checkpoint_attn': False,
                'use_checkpoint_ffn': False,
                'no_checkpoint_attn_blocks': [],
                'no_checkpoint_ffn_blocks': [],
                'init_type': 'default',
                'init_bn_type': 'uniform',
                'dcn_type': 'DCNv2'  # 将在测试中修改
            },
            'train': {
                'total_iter': 1,
                'manual_seed': 42
            }
        }

    def test_dcnv2_stage_creation(self, minimal_vrt_config):
        """测试DCNv2 Stage创建"""
        from models.architectures.vrt.stages import Stage

        config = minimal_vrt_config.copy()
        config['netG']['dcn_type'] = 'DCNv2'

        stage = Stage(
            in_dim=32,
            dim=32,
            input_resolution=(64, 64),
            depth=1,
            num_heads=2,
            window_size=(8, 8, 8),
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            drop_path=0.0,
            norm_layer=torch.nn.LayerNorm,
            pa_frames=2,
            deformable_groups=4,
            max_residue_magnitude=10,
            use_checkpoint_attn=False,
            use_checkpoint_ffn=False,
            use_sgp=False,
            reshape='none',
            sgp_w=1,
            sgp_k=3,
            sgp_reduction=8,
            dcn_config={
                'type': config['netG']['dcn_type'],
                'apply_softmax': config['netG'].get('dcn_apply_softmax', False),
            }
        )

        assert stage is not None
        assert hasattr(stage, 'pa_deform')
        assert stage.pa_deform.__class__.__name__ == 'DCNv2PackFlowGuided'

    def test_dcnv4_stage_creation(self, minimal_vrt_config):
        """测试DCNv4 Stage创建"""
        from models.architectures.vrt.stages import Stage

        config = minimal_vrt_config.copy()
        config['netG']['dcn_type'] = 'DCNv4'

        stage = Stage(
            in_dim=32,
            dim=32,
            input_resolution=(64, 64),
            depth=1,
            num_heads=2,
            window_size=(8, 8, 8),
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            drop_path=0.0,
            norm_layer=torch.nn.LayerNorm,
            pa_frames=2,
            deformable_groups=4,
            max_residue_magnitude=10,
            use_checkpoint_attn=False,
            use_checkpoint_ffn=False,
            use_sgp=False,
            reshape='none',
            sgp_w=1,
            sgp_k=3,
            sgp_reduction=8,
            dcn_config={
                'type': config['netG']['dcn_type'],
                'apply_softmax': config['netG'].get('dcn_apply_softmax', False),
            }
        )

        assert stage is not None
        assert hasattr(stage, 'pa_deform')
        assert stage.pa_deform.__class__.__name__ == 'DCNv4PackFlowGuided'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_dcnv4_stage_forward(self, minimal_vrt_config):
        """测试DCNv4 Stage前向传播（需要CUDA）"""
        from models.architectures.vrt.stages import Stage

        config = minimal_vrt_config.copy()
        config['netG']['dcn_type'] = 'DCNv4'

        stage = Stage(
            in_dim=32,
            dim=32,
            input_resolution=(64, 64),
            depth=1,
            num_heads=2,
            window_size=(8, 8, 8),
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            drop_path=0.0,
            norm_layer=torch.nn.LayerNorm,
            pa_frames=2,
            deformable_groups=4,
            max_residue_magnitude=10,
            use_checkpoint_attn=False,
            use_checkpoint_ffn=False,
            use_sgp=False,
            reshape='none',
            sgp_w=1,
            sgp_k=3,
            sgp_reduction=8,
            dcn_config={
                'type': config['netG']['dcn_type'],
                'apply_softmax': config['netG'].get('dcn_apply_softmax', False),
            }
        ).cuda()

        # 创建测试输入 (VRT Stage期望格式: [num_frames, channels, depth, height, width])
        num_frames, channels, depth, height, width = 4, 32, 1, 64, 64
        x = torch.randn(num_frames, channels, depth, height, width).cuda()

        # 创建光流（模拟，VRT格式）
        flow_shape = (num_frames-1, 2, height, width)
        flows_backward = [torch.randn(*flow_shape).cuda()]
        flows_forward = [torch.randn(*flow_shape).cuda()]

        # 前向传播
        output = stage(x, flows_backward, flows_forward)

        assert output.shape == x.shape
        assert output.device.type == 'cuda'

    def test_dcn_factory_function(self, minimal_vrt_config):
        """测试DCN工厂函数"""
        from models.blocks.dcn import get_deformable_module

        # 测试DCNv2
        config = minimal_vrt_config.copy()
        config['netG']['dcn_type'] = 'DCNv2'
        dcnv2_creator = get_deformable_module(config)
        dcnv2_module = dcnv2_creator(
            32, 32, 3, padding=1, deformable_groups=4, max_residue_magnitude=10, pa_frames=2
        )
        assert dcnv2_module.__class__.__name__ == 'DCNv2PackFlowGuided'

        # 测试DCNv4
        config['netG']['dcn_type'] = 'DCNv4'
        dcnv4_creator = get_deformable_module(config)
        dcnv4_module = dcnv4_creator(
            32, 32, 3, padding=1, deformable_groups=4, max_residue_magnitude=10, pa_frames=2
        )
        assert dcnv4_module.__class__.__name__ == 'DCNv4PackFlowGuided'

        # 测试默认值
        config['netG'].pop('dcn_type', None)
        default_creator = get_deformable_module(config)
        default_module = default_creator(
            32, 32, 3, padding=1, deformable_groups=4, max_residue_magnitude=10, pa_frames=2
        )
        assert default_module.__class__.__name__ == 'DCNv2PackFlowGuided'

    def test_config_file_parsing(self):
        """测试配置文件中的dcn_type字段"""
        import json
        import os

        config_files = [
            'options/gopro_rgb_local.json',
            'options/gopro_rgbspike_local.json'
        ]

        for config_file in config_files:
            if os.path.exists(config_file):
                # 读取并验证JSON语法（跳过注释）
                with open(config_file, 'r') as f:
                    content = f.read()

                # 移除注释行
                lines = content.split('\n')
                json_lines = [line for line in lines if not line.strip().startswith('//')]
                json_content = '\n'.join(json_lines)

                # 验证JSON语法
                config = json.loads(json_content)

                # 验证netG中有dcn_type字段
                assert 'netG' in config
                assert 'dcn_type' in config['netG']
                assert config['netG']['dcn_type'] in ['DCNv2', 'DCNv4']

                print(f"✓ Config file {config_file} validated successfully")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


