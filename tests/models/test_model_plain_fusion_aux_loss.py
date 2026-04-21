"""Tests for ModelPlain._compute_fusion_aux_loss and phase-aware loss weighting."""
import torch
import pytest


def _make_opt(phase1_aux=1.0, phase2_aux=0.2, passthrough=0.2, fix_iter=10):
    return {
        'scale': 1,
        'n_channels': 3,
        'netG': {
            'net_type': 'vrt',
            'input': {'strategy': 'fusion', 'mode': 'dual', 'raw_ingress_chans': 7},
            'fusion': {
                'enable': True, 'placement': 'early', 'operator': 'gated',
                'out_chans': 3, 'operator_params': {},
            },
            'in_chans': 7, 'upscale': 1,
            'img_size': [6, 8, 8], 'window_size': [6, 8, 8],
            'depths': [2, 2, 2, 2, 2, 2, 2, 2], 'indep_reconsts': [6, 7], 'embed_dims': [16] * 8, 'num_heads': [2] * 8,
            'output_mode': 'restoration',
            'restoration_reducer': {'type': 'index', 'index': 2},
            'pa_frames': 2,
            'use_flash_attn': False,
            'optical_flow': {'module': 'spynet', 'checkpoint': None, 'params': {}},
            'deformable_groups': 4,
            'nonblind_denoising': False,
            'use_checkpoint_attn': False,
            'use_checkpoint_ffn': False,
            'no_checkpoint_attn_blocks': [],
            'no_checkpoint_ffn_blocks': [],
            'dcn_type': 'DCNv2',
            'dcn_apply_softmax': False,
            'init_type': 'default',
            'init_bn_type': 'uniform',
            'init_gain': 0.2,
        },
        'train': {
            'G_lossfn_type': 'charbonnier',
            'G_lossfn_weight': 1.0,
            'G_charbonnier_eps': 1e-6,
            'phase1_fusion_aux_loss_weight': phase1_aux,
            'phase2_fusion_aux_loss_weight': phase2_aux,
            'fusion_passthrough_loss_weight': passthrough,
            'G_optimizer_type': 'adam',
            'G_optimizer_lr': 1e-4,
            'G_optimizer_betas': [0.9, 0.99],
            'G_optimizer_wd': 0,
            'G_optimizer_clipgrad': None,
            'G_optimizer_reuse': False,
            'G_scheduler_type': 'MultiStepLR',
            'G_scheduler_milestones': [100],
            'G_scheduler_gamma': 0.5,
            'G_regularizer_orthstep': None,
            'G_regularizer_clipstep': None,
            'G_param_strict': False,
            'E_param_strict': False,
            'E_decay': 0,
            'manual_seed': 0,
            'fix_iter': fix_iter,
            'fix_keys': [],
            'checkpoint_save': 100,
            'checkpoint_test': 100,
            'checkpoint_print': 10,
            'amp': {'enable': False},
            'freeze_backbone': False,
        },
        'path': {
            'root': '/tmp',
            'models': '/tmp',
            'pretrained_netG': None,
            'pretrained_netE': None,
            'pretrained_optimizerG': None,
        },
        'rank': 0,
        'dist': False,
        'is_train': True,
    }


def _inject_fusion_hook(model, fusion_out, spike_bins):
    bare = model.get_bare_model(model.netG)
    bare._last_fusion_out = fusion_out
    bare._last_spike_bins = spike_bins
    B = fusion_out.shape[0]
    N = fusion_out.shape[1] // spike_bins
    H, W = fusion_out.shape[-2], fusion_out.shape[-1]
    model.H = torch.zeros(B, N, 3, H, W)
    # self.L: [B, N, 3+spike_chans, H, W]; first 3 channels = blur_rgb
    model.L = torch.ones(B, N, 7, H, W)


def test_fusion_aux_loss_phase1_returns_nonzero():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt())
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)  # N=6, S=4
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() > 0.0


def test_fusion_aux_loss_phase2_smaller_than_phase1():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt(phase1_aux=1.0, phase2_aux=0.2))
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    loss_p1 = model._compute_fusion_aux_loss(is_phase1=True)
    loss_p2 = model._compute_fusion_aux_loss(is_phase1=False)
    assert loss_p2.item() < loss_p1.item()


def test_fusion_aux_loss_zero_when_all_weights_zero():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt(phase1_aux=0.0, phase2_aux=0.0, passthrough=0.0))
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() == 0.0


def test_fusion_aux_loss_no_hook_returns_zero():
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt())
    model.define_loss()
    fusion_out = torch.randn(1, 24, 3, 8, 8)
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    del model.get_bare_model(model.netG)._last_fusion_out
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    assert loss.item() == 0.0


def test_fusion_aux_loss_center_frame_indexing():
    """S//2::S indexing must select frames 2,6,10,14,18,22 for S=4."""
    from models.model_plain import ModelPlain
    model = ModelPlain(_make_opt())
    model.define_loss()
    fusion_out = torch.zeros(1, 24, 3, 8, 8)
    fusion_out[:, 2::4, :, :, :] = 1.0   # center frames = 1.0
    _inject_fusion_hook(model, fusion_out, spike_bins=4)
    model.H = torch.ones(1, 6, 3, 8, 8)  # GT = 1.0
    loss = model._compute_fusion_aux_loss(is_phase1=True)
    # fusion_center == GT == 1.0 → Charbonnier loss at floor (sqrt(eps) per element)
    # floor = (aux_weight + pass_weight) * sqrt(eps) = 1.2 * sqrt(1e-6) ≈ 0.0012
    assert loss.item() < 0.002
