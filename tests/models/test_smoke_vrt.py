#!/usr/bin/env python3
import importlib
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def run_smoke():
    print('Starting smoke tests: old vs new VRT implementations')

    # small random batch (use larger spatial size to satisfy SpyNet pyramid)
    x = torch.randn(1, 6, 3, 128, 128)

    # import new implementation (we compare within the refactored codebase)
    new_mod = importlib.import_module('models.architectures.vrt.vrt')
    old_mod = new_mod  # legacy module removed; ensure tests use the refactored module

    OldVRT = old_mod.VRT
    NewVRT = new_mod.VRT

    params = dict(upscale=1,
                  in_chans=3,
                  out_chans=3,
                  img_size=[6, 128, 128],
                  window_size=[6, 8, 8],
                  depths=[1] * 13,
                  indep_reconsts=[11, 12],
                  embed_dims=[64] * 13,
                  num_heads=[1] * 13,
                  pa_frames=2)

    # reseed to ensure identical parameter initialization for fair comparison
    torch.manual_seed(0)
    np.random.seed(0)
    old_model = OldVRT(**params)
    torch.manual_seed(0)
    np.random.seed(0)
    new_model = NewVRT(**params)
    old_model.eval()
    new_model.eval()

    with torch.no_grad():
        out_old = old_model(x)
        out_new = new_model(x)

    print('old output shape:', tuple(out_old.shape))
    print('new output shape:', tuple(out_new.shape))

    # compute differences (broadcast to same shapes if needed)
    try:
        diff = (out_old - out_new).reshape(-1)
        l2 = float(torch.norm(diff).item())
        maxabs = float(diff.abs().max().item())
        print('L2 diff:', l2)
        print('max abs diff:', maxabs)
    except Exception as e:
        print('Could not compute elementwise diff:', e)

    # SpyNet forward checks
    print('\\nRunning SpyNet + flow_warp + aligned image checks')
    warp_new = importlib.import_module('models.architectures.vrt.warp')
    sp_old = warp_new.SpyNet()
    sp_new = warp_new.SpyNet()

    ref = x[:, 0, :, :, :].contiguous()
    supp = x[:, 1, :, :, :].contiguous()
    try:
        f_old = sp_old(ref, supp)
        f_new = sp_new(ref, supp)
        print('SpyNet old output type:', type(f_old))
        print('SpyNet new output type:', type(f_new))
    except Exception as e:
        print('SpyNet forward failed:', e)

    # flow_warp basic check
    from models.utils.flow import flow_warp
    dummy_flow = torch.zeros(1, 128, 128, 2)
    try:
        warped = flow_warp(ref, dummy_flow, interp_mode='bilinear')
        print('flow_warp output shape:', tuple(warped.shape))
    except Exception as e:
        print('flow_warp failed:', e)

    # get_aligned_image_2frames / get_flows
    try:
        flows_b_old, flows_f_old = old_model.get_flows(x)
        flows_b_new, flows_f_new = new_model.get_flows(x)
        print('get_flows succeeded. sample shapes (new):', [tuple(t.shape) for t in flows_b_new])
    except Exception as e:
        print('get_flows failed:', e)

    try:
        xb_new, xf_new = new_model.get_aligned_image_2frames(x, flows_b_new[0], flows_f_new[0])
        print('get_aligned_image_2frames (new) shapes:', tuple(xb_new.shape), tuple(xf_new.shape))
    except Exception as e:
        print('get_aligned_image_2frames failed:', e)

    # Stage.forward smoke
    print('\\nRunning Stage.forward smoke test (pa_frames=0 to avoid DCN)')
    from models.architectures.vrt.stages import Stage
    # Stage requires a reshape mode; use 'none' to keep dimensions
    stage = Stage(in_dim=64, dim=64, input_resolution=(6, 16, 16), depth=1, num_heads=1, window_size=[6,8,8], pa_frames=0, reshape='none')
    inp = torch.randn(1, 64, 6, 16, 16)
    try:
        out_stage = stage(inp, [], [])
        print('Stage output shape:', tuple(out_stage.shape))
    except Exception as e:
        print('Stage.forward failed:', e)

    print('\\nSmoke tests finished.')

if __name__ == '__main__':
    run_smoke()


