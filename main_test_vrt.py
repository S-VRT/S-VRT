# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import cv2
import glob
import os
import torch
import torch.distributed as dist
import requests
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from models.network_vrt import VRT as net
from utils import utils_image as util
from data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, \
    SingleVideoRecurrentTestDataset, VFI_DAVIS, VFI_UCF101, VFI_Vid4
from data.select_dataset import define_Dataset


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


def _load_inference_config(opt_path: str) -> dict:
    cfg_path = Path(opt_path)
    text = cfg_path.read_text(encoding='utf-8')
    return json.loads(_strip_json_comments(text))


def _normalize_triplet(values, arg_name):
    if values is None:
        raise ValueError(f'{arg_name} must be provided via CLI or JSON config.')
    if len(values) != 3:
        raise ValueError(f'Expected three values, got {values}')
    return [int(v) for v in values]


def _merge_config_into_args(args, cfg):
    datasets = cfg.get('datasets', {}) or {}
    test_cfg = datasets.get('test', {}) or {}
    val_cfg = cfg.get('val', {}) or {}
    path_cfg = cfg.get('path', {}) or {}
    netG_cfg = cfg.get('netG', {}) or {}

    args.task = args.task or val_cfg.get('task_name') or cfg.get('task')
    args.folder_lq = args.folder_lq or test_cfg.get('dataroot_lq')
    args.folder_gt = args.folder_gt or test_cfg.get('dataroot_gt')

    if args.sigma is None:
        cfg_sigma = test_cfg.get('sigma')
        if cfg_sigma is None:
            cfg_sigma = val_cfg.get('sigma')
        args.sigma = cfg_sigma

    if args.num_workers is None:
        args.num_workers = test_cfg.get('dataloader_num_workers')

    if args.tile is None:
        tile_temporal = val_cfg.get('num_frame_testing')
        tile_height = val_cfg.get('size_patch_testing')
        tile_width = val_cfg.get('size_patch_testing_w', tile_height)
        if tile_temporal is not None and tile_height is not None and tile_width is not None:
            args.tile = [tile_temporal, tile_height, tile_width]

    if args.tile_overlap is None:
        # match training-side default: temporal overlap=2, spatial overlap=20 when not provided
        overlap_temporal = val_cfg.get('num_frame_overlapping', 2)
        overlap_height = val_cfg.get('size_patch_overlapping', val_cfg.get('size_patch_overlap', 20))
        overlap_width = val_cfg.get('size_patch_overlapping_w', overlap_height)
        if overlap_temporal is not None and overlap_height is not None and overlap_width is not None:
            args.tile_overlap = [overlap_temporal, overlap_height, overlap_width]

    if val_cfg.get('save_img') and not args.save_result:
        args.save_result = True

    # Store config for model loading
    args.cfg = cfg
    args.path_cfg = path_cfg
    args.netG_cfg = netG_cfg
    args.test_cfg = test_cfg


def _finalize_inference_args(args):
    missing = []
    if not args.task:
        missing.append('--task or config val.task_name/task')
    if not args.folder_lq:
        missing.append('--folder_lq or datasets.test.dataroot_lq')
    if args.num_workers is None:
        missing.append('--num_workers or datasets.test.dataloader_num_workers')
    if args.tile is None:
        missing.append('val.num_frame_testing/size_patch_testing (--tile)')
    if args.tile_overlap is None:
        missing.append('val.num_frame_overlapping/size_patch_overlapping (--tile_overlap)')

    if missing:
        raise ValueError('Missing required testing parameters: ' + ', '.join(missing))

    args.tile = _normalize_triplet(args.tile, 'tile')
    args.tile_overlap = _normalize_triplet(args.tile_overlap, 'tile_overlap')

    if args.folder_lq:
        args.folder_lq = osp.expanduser(args.folder_lq)
    if args.folder_gt:
        args.folder_gt = osp.expanduser(args.folder_gt)


def _init_device():
    '''Select device respecting LOCAL_RANK for torchrun.'''
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return torch.device(f'cuda:{local_rank}')
    return torch.device('cpu')


def _init_distributed():
    '''Initialize torch distributed if launched with torchrun.'''
    if dist.is_available() and 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank() if dist.is_initialized() else 0
        return True, rank, world_size
    return False, 0, 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=None,
                        help='Path to config JSON (same as training); overrides CLI defaults.')
    parser.add_argument('--task', type=str, default=None, help='tasks: 001 to 008')
    parser.add_argument('--sigma', type=int, default=None, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--folder_lq', type=str, default=None,
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs=3, metavar=('T', 'H', 'W'), default=None,
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs=3, metavar=('T', 'H', 'W'), default=None,
                        help='Overlapping of different tiles')
    parser.add_argument('--num_workers', type=int, default=None, help='number of workers in data loading')
    parser.add_argument('--save_result', action='store_true', help='save resulting image')
    args = parser.parse_args()

    cfg = None
    if args.opt:
        cfg = _load_inference_config(args.opt)
        _merge_config_into_args(args, cfg)

    _finalize_inference_args(args)

    # define model
    device = _init_device()
    is_dist, rank, world_size = _init_distributed()
    args.rank = rank
    model = prepare_model_dataset(args)
    model.eval()
    model = model.to(device)
    if 'vimeo' in args.folder_lq.lower():
        if 'videofi' in args.task:
            test_set = VideoTestVimeo90KDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_gt,
                                               'meta_info_file': "data/meta_info/meta_info_Vimeo90K_test_GT.txt",
                                                'pad_sequence': False, 'num_frame': 7, 'temporal_scale': 2,
                                                 'cache_data': False})
        else:
            test_set = VideoTestVimeo90KDataset({'dataroot_gt': args.folder_gt, 'dataroot_lq': args.folder_lq,
                                                 'meta_info_file': "data/meta_info/meta_info_Vimeo90K_test_GT.txt",
                                                 'pad_sequence': True, 'num_frame': 7, 'temporal_scale': 1,
                                                 'cache_data': False})
    elif 'davis' in args.folder_lq.lower() and 'videofi' in args.task:
        test_set = VFI_DAVIS(data_root=args.folder_gt)
    elif 'ucf101' in args.folder_lq.lower() and 'videofi' in args.task:
        test_set = VFI_UCF101(data_root=args.folder_gt)
    elif 'vid4' in args.folder_lq.lower() and 'videofi' in args.task:
        test_set = VFI_Vid4(data_root=args.folder_gt)
    elif args.folder_gt is not None:
        # Use dataset_type from config if available, otherwise fall back to default
        if hasattr(args, 'test_cfg') and args.test_cfg and 'dataset_type' in args.test_cfg:
            # Build dataset options from config and args
            dataset_opt = dict(args.test_cfg)
            dataset_opt.update({
                'dataroot_gt': args.folder_gt,
                'dataroot_lq': args.folder_lq,
                'sigma': args.sigma,
                'num_frame': -1,
                'cache_data': False
            })
            test_set = define_Dataset(dataset_opt)
        else:
            test_set = VideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                                  'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})
    else:
        test_set = SingleVideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                              'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})

    test_batch_size = 1
    if hasattr(args, 'test_cfg') and args.test_cfg:
        bs_from_cfg = args.test_cfg.get('dataloader_batch_size')
        if bs_from_cfg is not None:
            test_batch_size = bs_from_cfg

    test_sampler = None
    if is_dist:
        test_sampler = DistributedSampler(
            test_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    test_loader = DataLoader(
        dataset=test_set,
        num_workers=args.num_workers,
        batch_size=test_batch_size,
        shuffle=False if test_sampler is None else False,
        drop_last=False,
        sampler=test_sampler,
    )

    save_dir = f'results/{args.task}'
    if args.save_result:
        os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'

    for idx, batch in enumerate(test_loader):
        lq = batch['L'].to(device)
        folder = batch['folder']
        gt = batch['H'] if 'H' in batch else None

        # inference
        with torch.no_grad():
            output = test_video(lq, model, args)

        if 'videofi' in args.task:
            output = output[:, :1, ...]
            batch['lq_path'] = batch['gt_path']
        elif 'videosr' in args.task and 'vimeo' in args.folder_lq.lower():
            output = output[:, 3:4, :, :, :]
            batch['lq_path'] = batch['gt_path']

        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        test_results_folder['psnr_y'] = []
        test_results_folder['ssim_y'] = []

        B, T = output.shape[0], output.shape[1]
        for b in range(B):
            for t in range(T):
                # save image
                img = output[b, t, ...].data.float().cpu().clamp_(0, 1).numpy()
                if img.ndim == 3:
                    img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
                img = (img * 255.0).round().astype(np.uint8)
                if args.save_result and img.ndim in (2, 3) and min(img.shape[:2]) > 0:
                    seq_name = None
                    try:
                        seq_name = osp.basename(batch['lq_path'][b][t]).split('.')[0]
                    except Exception:
                        seq_name = f'clip_{b:03d}_t{t:03d}'
                    os.makedirs(f'{save_dir}/{folder[b]}', exist_ok=True)
                    cv2.imwrite(f'{save_dir}/{folder[b]}/{seq_name}.png', img)

                # evaluate psnr/ssim
                if gt is not None:
                    img_gt = gt[b, t, ...].data.float().cpu().clamp_(0, 1).numpy()
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))
                    img_gt = (img_gt * 255.0).round().astype(np.uint8)
                    img_gt = np.squeeze(img_gt)

                    # validate shape before metrics to avoid errors
                    if img.shape != img_gt.shape or img.ndim not in (2, 3) or img_gt.ndim not in (2, 3) or min(img.shape[:2]) == 0:
                        if args.rank == 0:
                            print(f"[warn] skip metric for {folder[b]} b{b} t{t}: shape mismatch img{img.shape} gt{img_gt.shape}")
                        continue

                    test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                    if img_gt.ndim == 3:  # RGB image
                        img_y = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                        img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                        test_results_folder['psnr_y'].append(util.calculate_psnr(img_y, img_gt_y, border=0))
                        test_results_folder['ssim_y'].append(util.calculate_ssim(img_y, img_gt_y, border=0))
                    else:
                        test_results_folder['psnr_y'].append(test_results_folder['psnr'][-1])
                        test_results_folder['ssim_y'].append(test_results_folder['ssim'][-1])

        if gt is not None and len(test_results_folder['psnr']) > 0:
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
            print('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                      format(folder[0] if isinstance(folder, (list, tuple)) else folder, idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
        elif gt is not None and len(test_results_folder['psnr']) == 0 and args.rank == 0:
            print(f"[warn] metrics skipped for {folder[0] if isinstance(folder, (list, tuple)) else folder} ({idx}/{len(test_loader)}) due to empty valid frames")
        else:
            print('Testing {:20s}  ({:2d}/{})'.format(folder[0] if isinstance(folder, (list, tuple)) else folder, idx, len(test_loader)))

    # summarize psnr/ssim (distributed-aware, align with training val logic)
    if gt is not None:
        local_psnr_sum = sum(test_results['psnr'])
        local_ssim_sum = sum(test_results['ssim'])
        local_psnr_y_sum = sum(test_results['psnr_y'])
        local_ssim_y_sum = sum(test_results['ssim_y'])

        local_psnr_count = len(test_results['psnr'])
        local_ssim_count = len(test_results['ssim'])
        local_psnr_y_count = len(test_results['psnr_y'])
        local_ssim_y_count = len(test_results['ssim_y'])

        metrics_tensor = torch.tensor(
            [local_psnr_sum, local_ssim_sum, local_psnr_y_sum, local_ssim_y_sum,
             local_psnr_count, local_ssim_count, local_psnr_y_count, local_ssim_y_count],
            dtype=torch.float64, device=device)

        if is_dist:
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        (global_psnr_sum, global_ssim_sum, global_psnr_y_sum, global_ssim_y_sum,
         global_psnr_count, global_ssim_count, global_psnr_y_count,
         global_ssim_y_count) = metrics_tensor.tolist()

        global_psnr_count = int(round(global_psnr_count))
        global_ssim_count = int(round(global_ssim_count))
        global_psnr_y_count = int(round(global_psnr_y_count))
        global_ssim_y_count = int(round(global_ssim_y_count))

        if global_psnr_count > 0 and args.rank == 0:
            ave_psnr = global_psnr_sum / global_psnr_count
            ave_ssim = global_ssim_sum / max(global_ssim_count, 1)
            ave_psnr_y = global_psnr_y_sum / max(global_psnr_y_count, 1)
            ave_ssim_y = global_ssim_y_sum / max(global_ssim_y_count, 1)
            print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                  format(save_dir, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))
        elif args.rank == 0:
            print('[warn] no valid frames for PSNR/SSIM; all metrics skipped')


def prepare_model_dataset(args):
    ''' prepare model and dataset according to args.task. '''

    # Check if we should load model from config file
    if hasattr(args, 'netG_cfg') and args.netG_cfg and args.netG_cfg.get('net_type') == 'vrt':
        netG_cfg = args.netG_cfg
        path_cfg = args.path_cfg or {}
        pretrained_netG = path_cfg.get('pretrained_netG')
        pretrained_netE = path_cfg.get('pretrained_netE')
        
        # Build model from config (even if pretrained paths are not set, we need the model structure)
        model = net(upscale=netG_cfg.get('upscale', 1),
                   in_chans=netG_cfg.get('in_chans', 3),
                   img_size=netG_cfg['img_size'],
                   window_size=netG_cfg['window_size'],
                   depths=netG_cfg['depths'],
                   indep_reconsts=netG_cfg.get('indep_reconsts', []),
                   embed_dims=netG_cfg['embed_dims'],
                   num_heads=netG_cfg['num_heads'],
                   spynet_path=netG_cfg.get('spynet_path'),
                   pa_frames=netG_cfg.get('pa_frames', 2),
                   deformable_groups=netG_cfg.get('deformable_groups', 16),
                   nonblind_denoising=netG_cfg.get('nonblind_denoising', False),
                   use_checkpoint_attn=netG_cfg.get('use_checkpoint_attn', False),
                   use_checkpoint_ffn=netG_cfg.get('use_checkpoint_ffn', False),
                   no_checkpoint_attn_blocks=netG_cfg.get('no_checkpoint_attn_blocks', []),
                   no_checkpoint_ffn_blocks=netG_cfg.get('no_checkpoint_ffn_blocks', []),
                   use_sgp=netG_cfg.get('use_sgp', False),
                   sgp_w=netG_cfg.get('sgp_w', 3),
                   sgp_k=netG_cfg.get('sgp_k', 3),
                   sgp_reduction=netG_cfg.get('sgp_reduction', 4))
        
        # Set args for testing
        args.scale = netG_cfg.get('upscale', 1)
        args.window_size = netG_cfg['window_size']
        args.nonblind_denoising = netG_cfg.get('nonblind_denoising', False)
        
        # Load models: prioritize EMA (E) model if available, otherwise use G model
        model_loaded = False
        
        # First, try to load EMA (E) model if specified (EMA models are preferred for inference)
        if pretrained_netE and os.path.exists(pretrained_netE):
            print(f'Loading Encoder (EMA) model from {pretrained_netE}')
            pretrained_model_E = torch.load(pretrained_netE, map_location='cpu')
            # EMA models use 'params_ema' key
            if isinstance(pretrained_model_E, dict):
                if 'params_ema' in pretrained_model_E:
                    model.load_state_dict(pretrained_model_E['params_ema'], strict=False)
                    model_loaded = True
                    print('Encoder (EMA) model loaded successfully (using params_ema)')
                elif 'params' in pretrained_model_E:
                    model.load_state_dict(pretrained_model_E['params'], strict=False)
                    model_loaded = True
                    print('Encoder (EMA) model loaded successfully (using params)')
                else:
                    # Try to load as state dict directly
                    model.load_state_dict(pretrained_model_E, strict=False)
                    model_loaded = True
                    print('Encoder (EMA) model loaded successfully (direct state_dict)')
            else:
                model.load_state_dict(pretrained_model_E, strict=False)
                model_loaded = True
                print('Encoder (EMA) model loaded successfully (direct)')
        
        # If EMA model not available, load G model
        if not model_loaded and pretrained_netG and os.path.exists(pretrained_netG):
            print(f'Loading Generator model from {pretrained_netG}')
            pretrained_model = torch.load(pretrained_netG, map_location='cpu')
            # Handle different checkpoint formats
            if isinstance(pretrained_model, dict):
                if 'params' in pretrained_model:
                    model.load_state_dict(pretrained_model['params'], strict=False)
                elif 'params_ema' in pretrained_model:
                    model.load_state_dict(pretrained_model['params_ema'], strict=False)
                elif 'model_state_dict' in pretrained_model:
                    model.load_state_dict(pretrained_model['model_state_dict'], strict=False)
                else:
                    # Try to load as state dict directly
                    model.load_state_dict(pretrained_model, strict=False)
            else:
                model.load_state_dict(pretrained_model, strict=False)
            model_loaded = True
            print('Generator model loaded successfully')
        
        if not model_loaded:
            print(f'Warning: No model loaded. Check paths:')
            if pretrained_netE:
                print(f'  pretrained_netE: {pretrained_netE} (exists: {os.path.exists(pretrained_netE)})')
            if pretrained_netG:
                print(f'  pretrained_netG: {pretrained_netG} (exists: {os.path.exists(pretrained_netG)})')
            raise RuntimeError('No valid model checkpoint found. Please set pretrained_netE or pretrained_netG in config file.')
        
        return model

    # define model
    if args.task == '001_VRT_videosr_bi_REDS_6frames':
        model = net(upscale=4, img_size=[6,64,64], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task == '002_VRT_videosr_bi_REDS_16frames':
        model = net(upscale=4, img_size=[16,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=6, deformable_groups=24)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [8,8,8]
        args.nonblind_denoising = False

    elif args.task in ['003_VRT_videosr_bi_Vimeo_7frames', '004_VRT_videosr_bd_Vimeo_7frames']:
        model = net(upscale=4, img_size=[8,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=4, deformable_groups=16)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [8,8,8]
        args.nonblind_denoising = False

    elif args.task in ['005_VRT_videodeblurring_DVD']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task in ['006_VRT_videodeblurring_GoPro']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['GoPro11-part1', 'GoPro11-part2']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task in ['007_VRT_videodeblurring_REDS']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['REDS4']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task == '008_VRT_videodenoising_DAVIS':
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16,
                    nonblind_denoising=True)
        datasets = ['Set8', 'DAVIS-test']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = True

    elif args.task == '009_VRT_videofi_Vimeo_4frames':
        model = net(upscale=1, out_chans=3, img_size=[4,192,192], window_size=[4,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=0)
        datasets = ['UCF101', 'DAVIS-train']  # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 1
        args.window_size = [4,8,8]
        args.nonblind_denoising = False

    # download model
    model_path = f'model_zoo/vrt/{args.task}.pth'
    if os.path.exists(model_path):
        print(f'loading model from ./model_zoo/vrt/{model_path}')
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/{}'.format(os.path.basename(model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {model_path}')
        open(model_path, 'wb').write(r.content)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model, strict=True)

    # download datasets
    if os.path.exists(f'{args.folder_lq}'):
        print(f'using dataset from {args.folder_lq}')
    else:
        if 'vimeo' in args.folder_lq.lower():
            print(f'Vimeo dataset is not at {args.folder_lq}! Please refer to #training of Readme.md to download it.')
        else:
            os.makedirs('testsets', exist_ok=True)
            for dataset in datasets:
                url = f'https://github.com/JingyunLiang/VRT/releases/download/v0.0/testset_{dataset}.tar.gz'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading testing dataset {dataset}')
                open(f'testsets/{dataset}.tar.gz', 'wb').write(r.content)
                os.system(f'tar -xvf testsets/{dataset}.tar.gz -C testsets')
                os.system(f'rm testsets/{dataset}.tar.gz')

    return model


def test_video(lq, model, args):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = args.tile[0]
        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = args.scale
            num_frame_overlapping = args.tile_overlap[0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if args.nonblind_denoising else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = None
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = test_clip(lq_clip, model, args)
                if E is None:
                    _, _, c_out, _, _ = out_clip.size()
                    E = torch.zeros(b, d, c_out, h*sf, w*sf)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = args.window_size
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            output = test_clip(lq, model, args)
            output = output[:, :d_old, :, :, :]

        return output


def test_clip(lq, model, args):
    ''' test the clip as a whole or as patches. '''

    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, _, h, w = lq.size()
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        E = None
        W = None

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                out_patch = model(in_patch).detach().cpu()

                # Lazy-init accumulation buffers using model output shape (not LQ channels)
                if E is None:
                    _, _, c_out, _, _ = out_patch.size()
                    E = torch.zeros(b, d, c_out, h*sf, w*sf)
                    W = torch.zeros_like(E)

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output


if __name__ == '__main__':
    main()
