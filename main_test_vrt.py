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
import psutil  # 系统和进程信息
import gc  # 垃圾回收

from models.architectures.vrt import VRT as net
from utils import utils_image as util
from data.dataset_video_test import TestDataset, \
    SingleTestDataset
from data.select_dataset import define_Dataset
from models.fusion.debug import FusionDebugDumper


def get_memory_usage():
    """获取当前进程的内存使用情况"""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # 返回 RSS (Resident Set Size) 内存使用量，单位为 GB
        return mem_info.rss / (1024 ** 3)
    except Exception as e:
        return 0.0


def log_memory_stage(stage_name, rank=0):
    """记录内存使用阶段信息"""
    if rank == 0:
        mem_usage = get_memory_usage()
        print(f'[MEMORY] {stage_name} - Current memory usage: {mem_usage:.2f} GB')
        return mem_usage
    return 0.0


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


def _get_bare_model(model):
    return model.module if hasattr(model, 'module') else model


def dump_post_inference_fusion_debug(model, args, batch, batch_idx, dumper_cls=FusionDebugDumper):
    if not bool(getattr(args, 'fusion_debug', False)):
        return False
    max_batches = int(getattr(args, 'fusion_debug_max_batches', 1) or 1)
    if batch_idx >= max_batches:
        return False
    if int(getattr(args, 'rank', 0)) != 0:
        return False

    cfg = getattr(args, 'cfg', None)
    if not isinstance(cfg, dict):
        return False

    debug_opt = json.loads(json.dumps(cfg))
    debug_cfg = debug_opt.setdefault('netG', {}).setdefault('fusion', {}).setdefault('debug', {})
    debug_cfg['enable'] = True
    debug_cfg['save_images'] = True
    if getattr(args, 'fusion_debug_subdir', None):
        debug_cfg['subdir'] = args.fusion_debug_subdir
    else:
        debug_cfg.setdefault('subdir', 'fusion_post_infer')
    if getattr(args, 'fusion_debug_source_view', None):
        debug_cfg['source_view'] = args.fusion_debug_source_view
    debug_opt.setdefault('path', {})
    debug_opt['path']['images'] = getattr(args, 'fusion_debug_dir', None) or debug_opt['path'].get('images', 'results')

    dumper = dumper_cls(debug_opt)
    if not (getattr(dumper, 'enabled', False) and getattr(dumper, 'save_images', False)):
        return False

    bare = _get_bare_model(model)
    return dumper.dump_tensor(
        fusion_main=getattr(bare, '_last_fusion_main', None),
        fusion_exec=getattr(bare, '_last_fusion_exec', None),
        fusion_meta=getattr(bare, '_last_fusion_meta', {}) or {},
        current_step=batch_idx,
        folder=batch.get('folder'),
        gt=batch.get('H'),
        rank=getattr(args, 'rank', 0),
        lq_paths=batch.get('lq_path'),
    )


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
    parser.add_argument('--fusion_debug', action='store_true',
                        help='Dump fusion module outputs during inference after loading checkpoint weights.')
    parser.add_argument('--fusion_debug_dir', type=str, default=None,
                        help='Override root directory for fusion debug images.')
    parser.add_argument('--fusion_debug_subdir', type=str, default='fusion_post_infer',
                        help='Subdirectory under each folder for fusion debug images.')
    parser.add_argument('--fusion_debug_source_view', type=str, default=None, choices=['main', 'exec'],
                        help='Fusion debug view to dump: main fused output or execution/backbone view.')
    parser.add_argument('--fusion_debug_max_batches', type=int, default=1,
                        help='Maximum number of inference batches to dump for fusion debug.')
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

    print('[TEST] Starting VRT inference...')
    mem_before_model = log_memory_stage('Before model and dataset creation', rank)

    model = prepare_model_dataset(args)
    model.eval()
    model = model.to(device)

    mem_after_model = log_memory_stage('After model loaded to device', rank)
    if rank == 0:
        print(f'[TEST] Model creation completed. Memory delta: {mem_after_model - mem_before_model:.2f} GB')

    print('[TEST] Creating test dataset...')
    mem_before_dataset = log_memory_stage('Before creating test dataset', rank)

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
            test_set = TestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                                  'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})
    else:
        test_set = SingleTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                              'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})

    mem_after_dataset = log_memory_stage('After creating test dataset', rank)
    if rank == 0:
        print(f'[TEST] Dataset creation completed. Memory delta: {mem_after_dataset - mem_before_dataset:.2f} GB')
        print(f'[TEST] Test dataset uses cache_data=False - no in-memory caching')

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
    all_clip_results = []  # 存储所有clip的结果

    assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'

    mem_before_inference = log_memory_stage('Before starting inference loop', rank)
    if rank == 0:
        print(f'[TEST] Starting inference on {len(test_loader)} batches...')

    # Debug: check dataset properties
    print(f"Dataset length: {len(test_set)}")
    if hasattr(test_set, 'folders'):
        print(f"Available folders: {len(test_set.folders)}")
        if test_set.folders:
            print(f"First folder: {test_set.folders[0]}")

    # Get first sample to check shape (skip actual loading to avoid memory usage)
    if len(test_set) > 0:
        folder = test_set.folders[0]
        # Just check file paths without loading images
        if hasattr(test_set, 'imgs_lq') and folder in test_set.imgs_lq:
            lq_paths = test_set.imgs_lq[folder]
            gt_paths = test_set.imgs_gt[folder] if hasattr(test_set, 'imgs_gt') and folder in test_set.imgs_gt else []
            print(f"Sample folder: {folder}")
            print(f"LQ frames: {len(lq_paths)}")
            print(f"GT frames: {len(gt_paths)}")
            if lq_paths:
                print(f"First LQ path: {lq_paths[0]}")
            if gt_paths:
                print(f"First GT path: {gt_paths[0]}")
        else:
            print(f"Sample folder: {folder} (no cached path info available)")

    # Track local progress (per rank) in terms of video folders, not batches
    total_folders = len(test_sampler) if test_sampler is not None else len(test_set)
    processed_folders = 0

    for idx, batch in enumerate(test_loader):
        if idx == 0 and rank == 0:
            print(f'[TEST] Processing first batch - monitoring memory usage...')
        elif idx > 0 and idx % 10 == 0 and rank == 0:  # 每10个batch记录一次内存
            log_memory_stage(f'Processing batch {idx}/{len(test_loader)}', rank)
        lq = batch['L'].to(device)  # Shape: (batch_size, frames, channels, height, width)
        folder = batch['folder']
        gt = batch['H'] if 'H' in batch else None

        print(f"DEBUG main: Processing batch, lq shape: {lq.shape}")

        # inference - keep original test_video logic but ensure model compatibility
        with torch.no_grad():
            output = test_video(lq, model, args)
        if args.fusion_debug:
            dump_post_inference_fusion_debug(model, args, batch, idx)

        if 'videofi' in args.task:
            output = output[:, :1, ...]
            batch['lq_path'] = batch['gt_path']
        elif 'videosr' in args.task and 'vimeo' in args.folder_lq.lower():
            output = output[:, 3:4, :, :, :]
            batch['lq_path'] = batch['gt_path']

        B, T = output.shape[0], output.shape[1]
        for b in range(B):
            folder_name = folder[b] if isinstance(folder, (list, tuple)) else folder
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
                    os.makedirs(f'{save_dir}/{folder_name}', exist_ok=True)
                    cv2.imwrite(f'{save_dir}/{folder_name}/{seq_name}.png', img)

                # evaluate psnr/ssim
                if gt is not None:
                    img_gt = gt[b, t, ...].data.float().cpu().clamp_(0, 1).numpy()
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))
                    img_gt = (img_gt * 255.0).round().astype(np.uint8)
                    img_gt = np.squeeze(img_gt)

                    # validate shape before metrics to avoid errors
                    if img.shape != img_gt.shape or img.ndim not in (2, 3) or img_gt.ndim not in (2, 3) or min(img.shape[:2]) == 0:
                        print(f"[Rank {args.rank}] [warn] skip metric for {folder[b]} b{b} t{t}: shape mismatch img{img.shape} gt{img_gt.shape}")
                        continue

                    clip_psnr = util.calculate_psnr(img, img_gt, border=0)
                    clip_ssim = util.calculate_ssim(img, img_gt, border=0)
                    if img_gt.ndim == 3:  # RGB image
                        img_y = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                        img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                        clip_psnr_y = util.calculate_psnr(img_y, img_gt_y, border=0)
                        clip_ssim_y = util.calculate_ssim(img_y, img_gt_y, border=0)
                    else:
                        clip_psnr_y = clip_psnr
                        clip_ssim_y = clip_ssim

                    # 存储每个clip的结果
                    clip_result = {
                        'folder_name': folder_name,
                        'clip_name': f'clip_{b:03d}_t{t:03d}',
                        'psnr': clip_psnr,
                        'ssim': clip_ssim,
                        'psnr_y': clip_psnr_y,
                        'ssim_y': clip_ssim_y
                    }
                    all_clip_results.append(clip_result)

            processed_folders += 1
            print('[Rank {}] Testing {:20s}  ({:2d}/{})'.format(args.rank, folder_name, processed_folders, total_folders))

    # summarize psnr/ssim (distributed-aware, align with training val logic)
    if gt is not None:
        # Collect all folder results across ranks for rank 0 to print and compute global stats
        world_size = world_size if 'world_size' in locals() else (dist.get_world_size() if dist.is_initialized() else 1)

        # Create test_results from all_clip_results (group by folder)
        test_results = {'psnr': [], 'ssim': [], 'psnr_y': [], 'ssim_y': []}
        folder_results = {}

        # Group clips by folder and compute folder averages
        for clip_result in all_clip_results:
            folder_name = clip_result['folder_name']
            if folder_name not in folder_results:
                folder_results[folder_name] = {'psnr': [], 'ssim': [], 'psnr_y': [], 'ssim_y': []}
            folder_results[folder_name]['psnr'].append(clip_result['psnr'])
            folder_results[folder_name]['ssim'].append(clip_result['ssim'])
            folder_results[folder_name]['psnr_y'].append(clip_result['psnr_y'])
            folder_results[folder_name]['ssim_y'].append(clip_result['ssim_y'])

        # Compute folder averages
        for folder_name, folder_data in folder_results.items():
            if folder_data['psnr']:
                avg_psnr = sum(folder_data['psnr']) / len(folder_data['psnr'])
                avg_ssim = sum(folder_data['ssim']) / len(folder_data['ssim'])
                avg_psnr_y = sum(folder_data['psnr_y']) / len(folder_data['psnr_y'])
                avg_ssim_y = sum(folder_data['ssim_y']) / len(folder_data['ssim_y'])

                test_results['psnr'].append(avg_psnr)
                test_results['ssim'].append(avg_ssim)
                test_results['psnr_y'].append(avg_psnr_y)
                test_results['ssim_y'].append(avg_ssim_y)

                # Print folder results (keep original behavior)
                print('[Rank {}] Testing {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                      format(args.rank, folder_name, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))

        # Now aggregate across ranks
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

        if is_dist and dist.is_initialized():
            # Gather per-rank folder metrics
            gathered = [torch.zeros_like(metrics_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, metrics_tensor)

            if args.rank == 0:
                # Print per-rank average results and collect all folder metrics for global max/avg
                all_folder_psnr = []
                all_folder_ssim = []
                all_folder_psnr_y = []
                all_folder_ssim_y = []

                for r, t in enumerate(gathered):
                    (psnr_sum, ssim_sum, psnr_y_sum, ssim_y_sum,
                     psnr_cnt, ssim_cnt, psnr_y_cnt, ssim_y_cnt) = t.tolist()
                    psnr_cnt = int(round(psnr_cnt))
                    ssim_cnt = int(round(ssim_cnt))
                    psnr_y_cnt = int(round(psnr_y_cnt))
                    ssim_y_cnt = int(round(ssim_y_cnt))
                    if psnr_cnt > 0:
                        ave_psnr_r = psnr_sum / psnr_cnt
                        ave_ssim_r = ssim_sum / max(ssim_cnt, 1)
                        ave_psnr_y_r = psnr_y_sum / max(psnr_y_cnt, 1)
                        ave_ssim_y_r = ssim_y_sum / max(ssim_y_cnt, 1)
                        print('[Rank {}] Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                              format(r, ave_psnr_r, ave_ssim_r, ave_psnr_y_r, ave_ssim_y_r))

                        # Collect all folder metrics for global max/avg calculation
                        all_folder_psnr.extend([ave_psnr_r] * psnr_cnt)
                        all_folder_ssim.extend([ave_ssim_r] * ssim_cnt)
                        all_folder_psnr_y.extend([ave_psnr_y_r] * psnr_y_cnt)
                        all_folder_ssim_y.extend([ave_ssim_y_r] * ssim_y_cnt)

                # Compute global maximums and averages
                if all_folder_psnr:
                    # Find the clip with maximum PSNR and use its complete metrics
                    max_psnr_idx = all_folder_psnr.index(max(all_folder_psnr))
                    max_psnr = all_folder_psnr[max_psnr_idx]
                    max_ssim = all_folder_ssim[max_psnr_idx]
                    max_psnr_y = all_folder_psnr_y[max_psnr_idx]
                    max_ssim_y = all_folder_ssim_y[max_psnr_idx]

                    avg_psnr = sum(all_folder_psnr) / len(all_folder_psnr)
                    avg_ssim = sum(all_folder_ssim) / len(all_folder_ssim)
                    avg_psnr_y = sum(all_folder_psnr_y) / len(all_folder_psnr_y)
                    avg_ssim_y = sum(all_folder_ssim_y) / len(all_folder_ssim_y)

                    print('\n{} \n-- Max PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                          format(save_dir, max_psnr, max_ssim, max_psnr_y, max_ssim_y))

                    print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                          format(save_dir, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))
        else:
            # Non-distributed: just compute local max and average
            if args.rank == 0:
                if test_results['psnr']:
                    # Find the clip with maximum PSNR and use its complete metrics
                    max_psnr_idx = test_results['psnr'].index(max(test_results['psnr']))
                    max_psnr = test_results['psnr'][max_psnr_idx]
                    max_ssim = test_results['ssim'][max_psnr_idx]
                    max_psnr_y = test_results['psnr_y'][max_psnr_idx]
                    max_ssim_y = test_results['ssim_y'][max_psnr_idx]

                    avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                    avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                    avg_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    avg_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

                    print('\n{} \n-- Max PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                          format(save_dir, max_psnr, max_ssim, max_psnr_y, max_ssim_y))

                    print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                          format(save_dir, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))

    mem_after_inference = log_memory_stage('After inference completion', rank)
    if rank == 0:
        print(f'[TEST] Inference completed. Total memory delta: {mem_after_inference - mem_before_inference:.2f} GB')
        print(f'[TEST] Final memory usage: {mem_after_inference:.2f} GB')


def _assert_lq_channels(tensor, tensor_name, netG_cfg):
    """Validate that the incoming temporal tensor matches configured in_chans."""
    expected_in_chans = netG_cfg.get('in_chans', 3)
    actual_in_chans = tensor.size(2)
    if actual_in_chans != expected_in_chans:
        raise ValueError(
            f"{tensor_name} Channel Mismatch!\n"
            f"Tensor shape: {tensor.shape} (Channels: {actual_in_chans})\n"
            f"Expected netG.in_chans: {expected_in_chans}\n"
            f"Mode: Test/Inference\n"
            f"Hint: Ensure your dataset returns all expected channels "
            f"(e.g., RGB 3 + Spike 4 = 7) before feeding them to netG."
        )


def prepare_model_dataset(args):
    ''' prepare model and dataset according to args.task. '''

    # Check if we should load model from config file
    # Prioritize config-based loading if netG_cfg is available and valid
    if hasattr(args, 'netG_cfg') and args.netG_cfg and isinstance(args.netG_cfg, dict) and args.netG_cfg.get('net_type') == 'vrt':
        netG_cfg = args.netG_cfg
        path_cfg = args.path_cfg or {}
        pretrained_netG = path_cfg.get('pretrained_netG')
        pretrained_netE = path_cfg.get('pretrained_netE')
        input_cfg = netG_cfg.get('input', {}) if isinstance(netG_cfg.get('input', {}), dict) else {}
        raw_ingress_chans = int(input_cfg.get('raw_ingress_chans', netG_cfg.get('in_chans', 3)))
        
        # Build model from config (even if pretrained paths are not set, we need the model structure)
        model = net(upscale=netG_cfg.get('upscale', 1),
                   in_chans=raw_ingress_chans,
                   img_size=netG_cfg['img_size'],
                   window_size=netG_cfg['window_size'],
                   depths=netG_cfg['depths'],
                   indep_reconsts=netG_cfg.get('indep_reconsts', []),
                   embed_dims=netG_cfg['embed_dims'],
                   num_heads=netG_cfg['num_heads'],
                   optical_flow=netG_cfg['optical_flow'],
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
                   sgp_reduction=netG_cfg.get('sgp_reduction', 4),
                   dcn_config={
                       'type': netG_cfg.get('dcn_type', 'DCNv2'),
                       'apply_softmax': netG_cfg.get('dcn_apply_softmax', False)
                   },
                   opt=getattr(args, 'cfg', None))
        
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
    # Config-based loading is preferred. If not available, raise an error
    if not hasattr(args, 'netG_cfg') or not args.netG_cfg:
        raise ValueError("Config-based model loading is required. "
                        "Please use --opt to specify a configuration file with a valid 'netG' section.")

    # Config-based model creation only - no more legacy hardcoded tasks

    # Data should be prepared via launch_test.sh --prepare-data or manually
    if not os.path.exists(f'{args.folder_lq}'):
        print(f'Warning: Dataset not found at {args.folder_lq}')
        print('Please ensure data is prepared using launch_test.sh --prepare-data or manually')
        print('For GoPro+Spike dataset, run: ./launch_test.sh --prepare-data')

    return model


def test_video(lq, model, args):
        '''test the video as a whole or as clips (divided temporally). '''

        # Channel validation - align with validation code
        netG_cfg = args.netG_cfg if hasattr(args, 'netG_cfg') and args.netG_cfg else {}
        _assert_lq_channels(lq, 'Test Video Input', netG_cfg)

        # Use tile-based testing as in original KAIR implementation

        # DEBUG: Print input shapes and configurations
        print(f"DEBUG test_video: input lq shape: {lq.shape}")
        print(f"DEBUG test_video: args.tile: {getattr(args, 'tile', 'None')}")
        print(f"DEBUG test_video: args.tile_overlap: {getattr(args, 'tile_overlap', 'None')}")
        print(f"DEBUG test_video: netG_cfg img_size: {netG_cfg.get('img_size', 'None')}")
        print(f"DEBUG test_video: netG_cfg in_chans: {netG_cfg.get('in_chans', 'None')}")

        # Use tile[0] for temporal tiling as in original logic
        num_frame_testing = args.tile[0] if args.tile and len(args.tile) > 0 else netG_cfg.get('img_size', [6, 160, 160])[0]
        print(f"DEBUG test_video: num_frame_testing: {num_frame_testing}")

        # Check if we have enough frames for tiling
        b, d, c, h, w = lq.size()
        print(f"DEBUG test_video: parsed shapes - b:{b}, d:{d}, c:{c}, h:{h}, w:{w}")
        c = c - 1 if args.nonblind_denoising else c

        # Ensure we have enough frames
        if d < num_frame_testing:
            print(f"WARNING: Video has {d} frames but model expects {num_frame_testing}. Using available frames.")
            num_frame_testing = d

        # Keep original behavior: use configured num_frame_testing, let model handle frame count issues
        if num_frame_testing > 0 and d >= num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = args.scale
            num_frame_overlapping = args.tile_overlap[0] if args.tile_overlap else 2
            not_overlap_border = False
            # Get output channels from config, similar to validation code
            c_out = netG_cfg.get('out_chans', c)

            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = None
            W = None

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = test_clip(lq_clip, model, args)
                if E is None:
                    # Use actual output channels from model, but fallback to config if needed
                    c_out = out_clip.size(2)
                    E = torch.zeros(b, d, c_out, h*sf, w*sf)
                    W = torch.zeros(b, d, 1, 1, 1)
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
            # Align padding logic with validation code: (d_old// window_size[0]+1)*window_size[0] - d_old
            window_size = args.window_size
            d_old = lq.size(1)
            d_pad = (d_old // window_size[0] + 1) * window_size[0] - d_old
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            output = test_clip(lq, model, args)
            output = output[:, :d_old, :, :, :]

        return output


def test_clip(lq, model, args):
    ''' test the clip as a whole or as patches. '''

    # Channel validation - align with validation code
    netG_cfg = args.netG_cfg if hasattr(args, 'netG_cfg') and args.netG_cfg else {}
    _assert_lq_channels(lq, 'Test Clip Input', netG_cfg)

    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        # Use overlap_size from config (validation code uses hardcoded 20, but test should use config)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        # Get output channels from config, similar to validation code
        c_out = netG_cfg.get('out_chans', c)
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
                    # Use actual output channels from model, but fallback to config if needed
                    c_out = out_patch.size(2)
                    E = torch.zeros(b, d, c_out, h*sf, w*sf)
                    W = torch.zeros(b, d, 1, h*sf, w*sf)

                out_patch_mask = torch.ones(b, d, 1, out_patch.size(-2), out_patch.size(-1))

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
        # Align padding logic with validation code: (h_old// window_size[1]+1)*window_size[1] - h_old
        _, _, _, h_old, w_old = lq.size()
        h_pad = (h_old // window_size[1] + 1) * window_size[1] - h_old
        w_pad = (w_old // window_size[2] + 1) * window_size[2] - w_old

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output


if __name__ == '__main__':
    main()
