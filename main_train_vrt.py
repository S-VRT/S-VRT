"""
VRT/RVRT 模型训练脚本
用于训练视频恢复任务（如视频去模糊、视频超分辨率等）的 VRT (Video Restoration Transformer) 模型
"""

import sys  # 系统相关功能，用于程序退出
import os.path  # 路径操作
import math  # 数学运算，用于计算迭代次数等
import argparse  # 命令行参数解析
import time  # 时间相关功能
import random  # 随机数生成，用于设置随机种子
import copy  # 深拷贝配置，避免阶段配置污染原始配置
import cv2  # OpenCV，用于图像读写
import numpy as np  # 数值计算库
from collections import OrderedDict  # 有序字典，用于保持测试结果的顺序
import logging  # 日志记录
import numbers
import torch  # PyTorch 深度学习框架
import torch.distributed as dist
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.data.distributed import DistributedSampler  # 分布式训练的数据采样器
import psutil  # 系统和进程信息
import gc  # 垃圾回收

# 工具函数导入
from utils import utils_logger  # 日志工具，包括 TensorBoard 和 WANDB 支持
from utils import utils_image as util  # 图像处理工具函数
from utils import utils_option as option  # 配置文件解析工具
from utils.utils_dist import get_dist_info, init_dist, barrier_safe, setup_distributed, get_rank, is_main_process  # 分布式训练工具
from utils.utils_profiler import TrainProfiler, TrainProfilerConfig
from utils.utils_runtime import apply_runtime_cpu_config
from utils.utils_two_run import (
    build_initial_two_run_state,
    dump_resolved_two_run_opts,
    load_two_run_state,
    mark_phase1_completed,
    mark_phase2_started,
    resolve_resume_phase,
    resolve_two_run_phase_opts,
    save_two_run_state,
    two_run_state_path,
    update_last_successful_step,
)

# 数据集和模型定义
from data.select_dataset import define_Dataset  # 数据集工厂函数
from models.model_vrt import should_score_validation_batch
from models.select_model import define_Model  # 模型工厂函数


class RepeatEpochDistributedSampler(DistributedSampler):
    def __init__(self, dataset, *, epoch_repeat=1, **kwargs):
        super().__init__(dataset, **kwargs)
        self.epoch_repeat = int(epoch_repeat)
        if self.epoch_repeat <= 0:
            raise ValueError(f"dataloader_epoch_repeat must be > 0, got {self.epoch_repeat}")
        self._base_num_samples = self.num_samples
        self._base_total_size = self.total_size
        self.num_samples = self._base_num_samples * self.epoch_repeat
        self.total_size = self._base_total_size * self.epoch_repeat

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self._base_total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[: self._base_total_size]

        indices = indices[self.rank : self._base_total_size : self.num_replicas]
        if self.epoch_repeat > 1:
            indices = indices * self.epoch_repeat
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"RepeatEpochDistributedSampler produced {len(indices)} samples, expected {self.num_samples}"
            )
        return iter(indices)


def resolve_phase_value(value, is_phase1, key_name):
    """Resolve scalar or [phase1, phase2] value to active phase value."""
    if isinstance(value, int):
        resolved = value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        resolved = value[0] if is_phase1 else value[1]
    else:
        raise ValueError(
            f"{key_name} must be an int or a length-2 list/tuple [phase1, phase2], got {value!r}"
        )

    if not isinstance(resolved, int):
        raise ValueError(f"{key_name} resolved value must be int, got {type(resolved).__name__}")
    if resolved <= 0:
        raise ValueError(f"{key_name} must be > 0, got {resolved}")
    return resolved


def build_phase_train_dataset_opt(train_dataset_opt, is_phase1):
    """Build per-phase dataset options with resolved gt_size and dataloader_batch_size."""
    resolved = copy.deepcopy(train_dataset_opt)
    resolved["gt_size"] = resolve_phase_value(train_dataset_opt.get("gt_size", 256), is_phase1, "gt_size")
    resolved["dataloader_batch_size"] = resolve_phase_value(
        train_dataset_opt["dataloader_batch_size"], is_phase1, "dataloader_batch_size"
    )
    if is_phase1:
        spike_flow_cfg = resolved.get("spike_flow")
        if isinstance(spike_flow_cfg, dict) and str(spike_flow_cfg.get("representation", "")).strip().lower() == "encoding25":
            spike_flow_cfg["representation"] = ""
            spike_flow_cfg["phase1_disabled"] = True
    return resolved


def compute_is_phase1(current_step, fix_iter):
    return fix_iter > 0 and current_step < fix_iter


def build_two_run_phase_model_train_opt(phase_train_opt, phase_name):
    resolved = copy.deepcopy(phase_train_opt)
    if phase_name == "phase1":
        fix_iter = int(resolved.get("fix_iter", 0) or 0)
        total_iter = int(resolved.get("total_iter", 0) or 0)
        if fix_iter > 0 and fix_iter == total_iter:
            resolved["fix_iter"] = total_iter + 1
    return resolved


def init_dataloader_worker(_worker_id):
    """Keep each DataLoader worker single-threaded to avoid CPU oversubscription."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def build_train_sampler(train_set, *, shuffle, seed, epoch_repeat=1, num_replicas=None, rank=None):
    kwargs = {
        "shuffle": shuffle,
        "drop_last": True,
        "seed": seed,
    }
    if num_replicas is not None:
        kwargs["num_replicas"] = num_replicas
    if rank is not None:
        kwargs["rank"] = rank
    return RepeatEpochDistributedSampler(
        train_set,
        epoch_repeat=int(epoch_repeat or 1),
        **kwargs,
    )


def build_timing_summary(logs, dist_enabled=False, device=None):
    """Add per-iteration timing max/mean fields across DDP ranks."""
    summarized = dict(logs)
    time_keys = [key for key in logs if key.startswith("time_")]
    if not time_keys:
        return summarized

    if dist_enabled and dist.is_available() and dist.is_initialized():
        values = torch.tensor([float(logs[key]) for key in time_keys], device=device or "cpu", dtype=torch.float64)
        max_values = values.clone()
        sum_values = values.clone()
        dist.all_reduce(max_values, op=dist.ReduceOp.MAX)
        dist.all_reduce(sum_values, op=dist.ReduceOp.SUM)
        mean_values = sum_values / float(dist.get_world_size())
        for key, max_value, mean_value in zip(time_keys, max_values.tolist(), mean_values.tolist()):
            summarized[f"{key}_max"] = max_value
            summarized[f"{key}_mean"] = mean_value
        return summarized

    for key in time_keys:
        summarized[f"{key}_max"] = float(logs[key])
        summarized[f"{key}_mean"] = float(logs[key])
    return summarized


def format_eta(seconds):
    if not isinstance(seconds, numbers.Real) or not math.isfinite(float(seconds)) or seconds <= 0:
        total_seconds = 0
    else:
        total_seconds = int(float(seconds))

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def compute_training_eta(current_step, total_iter, seconds_per_iter):
    remaining_iters = max(int(total_iter) - int(current_step), 0)
    return format_eta(remaining_iters * float(seconds_per_iter))


def build_train_loader_bundle(opt, train_dataset_opt, is_phase1, seed, logger):
    dataset_opt = build_phase_train_dataset_opt(train_dataset_opt, is_phase1)
    train_set = define_Dataset(dataset_opt)

    batch_size = dataset_opt["dataloader_batch_size"]
    if opt["dist"]:
        if batch_size % opt["num_gpu"] != 0:
            raise ValueError(
                f"dataloader_batch_size={batch_size} is not divisible by num_gpu={opt['num_gpu']}"
            )
        per_gpu_batch = batch_size // opt["num_gpu"]
        if per_gpu_batch <= 0:
            raise ValueError(
                f"per-GPU batch size must be > 0, got {per_gpu_batch} "
                f"(global batch={batch_size}, num_gpu={opt['num_gpu']})"
            )

        train_sampler = build_train_sampler(
            train_set,
            shuffle=dataset_opt["dataloader_shuffle"],
            seed=seed,
            epoch_repeat=dataset_opt.get("dataloader_epoch_repeat", 1),
            num_replicas=opt.get("num_gpu"),
            rank=opt.get("rank", 0),
        )
        per_gpu_workers = dataset_opt["dataloader_num_workers"] // opt["num_gpu"]
        kwargs = dict(
            batch_size=per_gpu_batch,
            shuffle=False,
            num_workers=per_gpu_workers,
            drop_last=True,
            pin_memory=True,
            sampler=train_sampler,
        )
        if per_gpu_workers > 0:
            kwargs["persistent_workers"] = dataset_opt.get("dataloader_persistent_workers", False)
            kwargs["prefetch_factor"] = dataset_opt.get("dataloader_prefetch_factor", 2)
            kwargs["multiprocessing_context"] = "spawn"
            kwargs["worker_init_fn"] = init_dataloader_worker
        train_loader = DataLoader(train_set, **kwargs)
    else:
        train_sampler = None
        workers = dataset_opt["dataloader_num_workers"]
        kwargs = dict(
            batch_size=batch_size,
            shuffle=dataset_opt["dataloader_shuffle"],
            num_workers=workers,
            drop_last=True,
            pin_memory=True,
        )
        if workers > 0:
            kwargs["persistent_workers"] = dataset_opt.get("dataloader_persistent_workers", False)
            kwargs["prefetch_factor"] = dataset_opt.get("dataloader_prefetch_factor", 2)
            kwargs["multiprocessing_context"] = "spawn"
            kwargs["worker_init_fn"] = init_dataloader_worker
        train_loader = DataLoader(train_set, **kwargs)

    return {
        "dataset_opt": dataset_opt,
        "train_set": train_set,
        "train_sampler": train_sampler,
        "train_loader": train_loader,
    }


def build_test_loader_bundle(opt, test_dataset_opt, seed, logger):
    if opt['rank'] == 0 and logger is not None:
        logger.info('[DATASET] Creating test/validation dataset...')
    mem_before_test = log_memory_stage(logger, 'Before creating test dataset', opt['rank'])

    test_set = define_Dataset(test_dataset_opt)

    mem_after_test = log_memory_stage(logger, 'After creating test dataset', opt['rank'])
    if opt['rank'] == 0 and logger is not None:
        logger.info('[DATASET] Test dataset created. Memory delta: {:.2f} GB'.format(mem_after_test - mem_before_test))
        logger.info('[DATASET] Test dataset does not use in-memory caching (cache_data=False by default)')

    test_batch_size = test_dataset_opt.get('dataloader_batch_size', 1)
    test_num_workers = test_dataset_opt.get('dataloader_num_workers', 1)
    test_shuffle = test_dataset_opt.get('dataloader_shuffle', False)

    if opt['dist']:
        if dist.is_initialized():
            test_sampler = DistributedSampler(
                test_set,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=test_shuffle,
                drop_last=False,
                seed=seed,
            )
        else:
            test_sampler = DistributedSampler(test_set, shuffle=test_shuffle, drop_last=False, seed=seed)
        per_gpu_batch_size = max(1, test_batch_size // opt['num_gpu'])
        per_gpu_num_workers = max(1, test_num_workers // opt['num_gpu'])
        test_loader_kwargs = dict(
            batch_size=per_gpu_batch_size,
            shuffle=False,
            num_workers=per_gpu_num_workers,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler,
        )
        if per_gpu_num_workers > 0:
            test_loader_kwargs['persistent_workers'] = test_dataset_opt.get('dataloader_persistent_workers', False)
            test_loader_kwargs['prefetch_factor'] = test_dataset_opt.get('dataloader_prefetch_factor', 2)
            test_loader_kwargs['multiprocessing_context'] = 'spawn'
        test_loader = DataLoader(test_set, **test_loader_kwargs)
    else:
        test_loader_kwargs = dict(
            batch_size=test_batch_size,
            shuffle=test_shuffle,
            num_workers=test_num_workers,
            drop_last=False,
            pin_memory=True,
        )
        if test_num_workers > 0:
            test_loader_kwargs['persistent_workers'] = test_dataset_opt.get('dataloader_persistent_workers', False)
            test_loader_kwargs['prefetch_factor'] = test_dataset_opt.get('dataloader_prefetch_factor', 2)
            test_loader_kwargs['multiprocessing_context'] = 'spawn'
        test_loader = DataLoader(test_set, **test_loader_kwargs)

    requested_dist_patch_val = bool(opt.get('val', {}).get('distributed_patch_testing', False))
    dist_world_size = dist.get_world_size() if opt['dist'] and dist.is_initialized() else opt.get('world_size', 1)
    active_dist_patch_val = requested_dist_patch_val and opt['dist'] and dist_world_size > 1 and len(test_set) == 1
    opt.setdefault('val', {})['distributed_patch_testing_active'] = active_dist_patch_val
    if requested_dist_patch_val and opt['rank'] == 0 and logger is not None:
        if active_dist_patch_val:
            logger.info(
                '[VALIDATION] Distributed patch testing active for single-sample validation '
                f'(world_size={dist_world_size}).'
            )
        else:
            logger.info(
                '[VALIDATION] Distributed patch testing requested but inactive '
                f'(dist={opt["dist"]}, world_size={dist_world_size}, test_set_size={len(test_set)}).'
            )

    return {
        "test_set": test_set,
        "test_loader": test_loader,
    }


def get_memory_usage():
    """获取当前进程的内存使用情况"""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # 返回 RSS (Resident Set Size) 内存使用量，单位为 GB
        return mem_info.rss / (1024 ** 3)
    except Exception as e:
        return 0.0


def log_memory_stage(logger, stage_name, rank=0):
    """记录内存使用阶段信息"""
    if rank == 0 and logger is not None:
        mem_usage = get_memory_usage()
        logger.info(f'[MEMORY] {stage_name} - Current memory usage: {mem_usage:.2f} GB')
        return mem_usage
    return 0.0


def log_validation_probe(logger, label, rank=0):
    """Log a compact validation-stage probe with process and CUDA memory state."""
    if rank != 0 or logger is None:
        return
    try:
        process = psutil.Process(os.getpid())
        rss_gb = process.memory_info().rss / (1024 ** 3)
        vms_gb = process.memory_info().vms / (1024 ** 3)
        cuda_parts = []
        if torch.cuda.is_available():
            for dev_idx in range(torch.cuda.device_count()):
                try:
                    alloc_gb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
                    reserved_gb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
                    cuda_parts.append(f'cuda:{dev_idx} alloc={alloc_gb:.2f}GB reserved={reserved_gb:.2f}GB')
                except Exception as cuda_exc:
                    cuda_parts.append(f'cuda:{dev_idx} error={cuda_exc}')
        cuda_summary = '; '.join(cuda_parts) if cuda_parts else 'cuda:unavailable'
        logger.info(f'[VAL_PROBE] {label} | rss={rss_gb:.2f}GB vms={vms_gb:.2f}GB | {cuda_summary}')
    except Exception as exc:
        logger.warning(f'[VAL_PROBE] {label} probe failed: {exc}')


def should_dump_full_frame_fusion_debug(model, current_step):
    """Return whether phase1 last-step validation full-frame debug is active."""
    dumper = getattr(model, 'fusion_debug', None)
    return (
        dumper is not None
        and dumper.should_dump_phase1_last(current_step, model.fix_iter, source='val_full_frame')
    )


def maybe_dump_full_frame_fusion_debug_from_batch(model, batch, current_step, batch_idx):
    """Reuse the first real validation batch for fusion debug to avoid a second loader pass."""
    if batch_idx != 0:
        return False
    if not should_dump_full_frame_fusion_debug(model, current_step):
        return False
    return model.dump_full_frame_fusion_only_from_batch(batch, current_step)


def should_run_validation_checkpoint(checkpoint_test, phase_step):
    if checkpoint_test in (None, 0, [], ()):
        return False
    if isinstance(checkpoint_test, list):
        return int(phase_step) in checkpoint_test
    return int(phase_step) % int(checkpoint_test) == 0


def run_validation_checkpoint(*, model, phase_opt, test_loader, tb_logger, logger, phase_step, global_step, epoch):
    opt = phase_opt
    if test_loader is None:
        return

    if opt['rank'] == 0 and logger is not None:
        logger.info('[VALIDATION] Starting model validation/test phase...')
    mem_before_validation = log_memory_stage(logger, 'Before validation memory cleanup', opt['rank'])

    is_master_process = opt['rank'] == 0
    if opt['dist']:
        barrier_safe()

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    gt = None

    for idx, test_data in enumerate(test_loader):
        if opt['rank'] == 0:
            maybe_dump_full_frame_fusion_debug_from_batch(model, test_data, phase_step, idx)
        if idx < 2:
            log_validation_probe(logger, f'before feed_data batch={idx}', opt['rank'])
            if opt['rank'] == 0 and logger is not None:
                batch_summary = []
                for key, value in test_data.items():
                    if hasattr(value, 'shape'):
                        batch_summary.append(f'{key}:{tuple(value.shape)}')
                    elif isinstance(value, (list, tuple)):
                        batch_summary.append(f'{key}:len={len(value)}')
                    else:
                        batch_summary.append(f'{key}:{type(value).__name__}')
                logger.info(f'[VAL_BATCH] idx={idx} contents: {", ".join(batch_summary)}')

        model.feed_data(test_data)
        if idx < 2:
            log_validation_probe(logger, f'after feed_data batch={idx}', opt['rank'])
        if idx < 2 and opt['rank'] == 0 and logger is not None:
            logger.info(f'[VAL_BATCH] idx={idx} entering model.test()')
        model.test()
        if idx < 2:
            log_validation_probe(logger, f'after model.test batch={idx}', opt['rank'])
            if opt['rank'] == 0 and logger is not None:
                logger.info(f'[VAL_BATCH] idx={idx} model.test() completed')

        if not should_score_validation_batch(opt):
            continue

        visuals = model.current_visuals()
        output = visuals['E']
        gt = visuals['H'] if 'H' in visuals else None
        folder = test_data['folder']
        folder_name = folder[0] if isinstance(folder, (list, tuple)) else folder

        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        test_results_folder['psnr_y'] = []
        test_results_folder['ssim_y'] = []

        lq_paths = test_data.get('lq_path')
        batch_clip_count = output.shape[0]
        for i in range(batch_clip_count):
            clip_name = f'clip_{i:03d}'
            if lq_paths is not None:
                clip_source = None
                try:
                    clip_source = lq_paths[i]
                except Exception:
                    clip_source = None
                if isinstance(clip_source, (list, tuple)) and clip_source:
                    clip_source = clip_source[0]
                if isinstance(clip_source, str):
                    clip_name = os.path.splitext(os.path.basename(clip_source))[0]

            img = output[i, ...].clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
            img = (img * 255.0).round().astype(np.uint8)

            if opt['val']['save_img']:
                save_dir = opt['path']['images']
                util.mkdir(save_dir)
                os.makedirs(f'{save_dir}/{folder_name}', exist_ok=True)
                cv2.imwrite(f'{save_dir}/{folder_name}/{clip_name}_{global_step:d}.png', img)

            if gt is not None:
                img_gt = gt[i, ...].clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))
                img_gt = (img_gt * 255.0).round().astype(np.uint8)
                img_gt = np.squeeze(img_gt)

                clip_psnr = util.calculate_psnr(img, img_gt, border=0)
                clip_ssim = util.calculate_ssim(img, img_gt, border=0)
                test_results_folder['psnr'].append(clip_psnr)
                test_results_folder['ssim'].append(clip_ssim)

                if img_gt.ndim == 3:
                    img_y = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                    img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                    clip_psnr_y = util.calculate_psnr(img_y, img_gt_y, border=0)
                    clip_ssim_y = util.calculate_ssim(img_y, img_gt_y, border=0)
                else:
                    clip_psnr_y = clip_psnr
                    clip_ssim_y = clip_ssim
                test_results_folder['psnr_y'].append(clip_psnr_y)
                test_results_folder['ssim_y'].append(clip_ssim_y)

        if len(test_results_folder['psnr']) > 0:
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)

            print('[Rank {}] Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                  format(opt['rank'], folder_name, len(test_results['psnr']), len(test_loader), psnr, ssim, psnr_y, ssim_y))

    local_psnr_sum = sum(test_results['psnr'])
    local_ssim_sum = sum(test_results['ssim'])
    local_psnr_y_sum = sum(test_results['psnr_y'])
    local_ssim_y_sum = sum(test_results['ssim_y'])

    local_psnr_count = len(test_results['psnr'])
    local_ssim_count = len(test_results['ssim'])
    local_psnr_y_count = len(test_results['psnr_y'])
    local_ssim_y_count = len(test_results['ssim_y'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics_tensor = torch.tensor(
        [local_psnr_sum, local_ssim_sum, local_psnr_y_sum, local_ssim_y_sum,
         local_psnr_count, local_ssim_count, local_psnr_y_count, local_ssim_y_count],
        dtype=torch.float64, device=device)

    world_size = opt.get('world_size', 1)

    if opt['dist'] and dist.is_initialized():
        gathered = [torch.zeros_like(metrics_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, metrics_tensor)

        if is_master_process and logger is not None:
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
                    logger.info('[Rank {}] Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                                format(r, ave_psnr_r, ave_ssim_r, ave_psnr_y_r, ave_ssim_y_r))

                    all_folder_psnr.extend([ave_psnr_r] * psnr_cnt)
                    all_folder_ssim.extend([ave_ssim_r] * ssim_cnt)
                    all_folder_psnr_y.extend([ave_psnr_y_r] * psnr_y_cnt)
                    all_folder_ssim_y.extend([ave_ssim_y_r] * ssim_y_cnt)

            if all_folder_psnr:
                max_psnr_idx = all_folder_psnr.index(max(all_folder_psnr))
                max_psnr = all_folder_psnr[max_psnr_idx]
                max_ssim = all_folder_ssim[max_psnr_idx]
                max_psnr_y = all_folder_psnr_y[max_psnr_idx]
                max_ssim_y = all_folder_ssim_y[max_psnr_idx]

                avg_psnr = sum(all_folder_psnr) / len(all_folder_psnr)
                avg_ssim = sum(all_folder_ssim) / len(all_folder_ssim)
                avg_psnr_y = sum(all_folder_psnr_y) / len(all_folder_psnr_y)
                avg_ssim_y = sum(all_folder_ssim_y) / len(all_folder_ssim_y)

                logger.info('<epoch:{:3d}, iter:{:8,d} Max PSNR: {:.2f} dB; SSIM: {:.4f}; '
                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                    epoch, global_step, max_psnr, max_ssim, max_psnr_y, max_ssim_y))

                logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                    epoch, global_step, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))

                if tb_logger is not None:
                    test_metrics = {
                        'psnr': max_psnr,
                        'ssim': max_ssim,
                        'psnr_y': max_psnr_y,
                        'ssim_y': max_ssim_y
                    }
                    tb_logger.log_scalars(global_step, test_metrics, tag_prefix='test')
    else:
        if is_master_process and logger is not None:
            if test_results['psnr']:
                max_psnr_idx = test_results['psnr'].index(max(test_results['psnr']))
                max_psnr = test_results['psnr'][max_psnr_idx]
                max_ssim = test_results['ssim'][max_psnr_idx]
                max_psnr_y = test_results['psnr_y'][max_psnr_idx]
                max_ssim_y = test_results['ssim_y'][max_psnr_idx]

                avg_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                avg_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                avg_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                avg_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

                logger.info('<epoch:{:3d}, iter:{:8,d} Max PSNR: {:.2f} dB; SSIM: {:.4f}; '
                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                    epoch, global_step, max_psnr, max_ssim, max_psnr_y, max_ssim_y))

                logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                    epoch, global_step, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))

                if tb_logger is not None:
                    test_metrics = {
                        'psnr': max_psnr,
                        'ssim': max_ssim,
                        'psnr_y': max_psnr_y,
                        'ssim_y': max_ssim_y
                    }
                    tb_logger.log_scalars(global_step, test_metrics, tag_prefix='test')

    mem_after_validation = log_memory_stage(logger, 'After validation completion', opt['rank'])
    if opt['rank'] == 0 and logger is not None:
        logger.info('[VALIDATION] Validation phase completed. Memory usage: {:.2f} GB'.format(mem_after_validation))

    if opt['dist']:
        barrier_safe()


def compute_global_step(global_step_offset, phase_step):
    return int(global_step_offset) + int(phase_step)


def should_finish_phase(current_phase_step, total_iter):
    return int(current_phase_step) >= int(total_iter)


def build_phase_runtime(phase_opt, seed, logger, phase_name):
    train_dataset_opt_base = phase_opt["datasets"]["train"]
    is_phase1 = phase_name == "phase1"
    bundle = build_train_loader_bundle(phase_opt, train_dataset_opt_base, is_phase1, seed, logger)
    test_bundle = {}
    test_dataset_opt = phase_opt.get("datasets", {}).get("test")
    if test_dataset_opt is not None:
        test_bundle = build_test_loader_bundle(phase_opt, test_dataset_opt, seed, logger)
    model_opt = copy.deepcopy(phase_opt)
    model_opt["train"] = build_two_run_phase_model_train_opt(
        model_opt.get("train", {}),
        phase_name=phase_name,
    )
    model = define_Model(model_opt)
    model.init_train()
    return {
        "model": model,
        "train_loader": bundle["train_loader"],
        "train_sampler": bundle["train_sampler"],
        "test_loader": test_bundle.get("test_loader"),
        "test_set": test_bundle.get("test_set"),
        "train_set": bundle["train_set"],
        "active_train_dataset_opt": bundle["dataset_opt"],
    }


def execute_training_iteration(*, model, train_data, phase_step, global_step):
    model.update_learning_rate(phase_step)
    model.feed_data(train_data, current_step=phase_step)
    model.optimize_parameters(phase_step)
    logs = model.current_log()
    logs["phase_step"] = float(phase_step)
    logs["global_step"] = float(global_step)
    return logs


def next_train_batch_with_timing(model, train_loader_iter):
    timer = getattr(getattr(model, "timer", None), "timer", None)
    if timer is None:
        return next(train_loader_iter)
    with timer("batch_wait"):
        return next(train_loader_iter)


def format_training_log_message(*, phase_name, epoch, phase_step, global_step, learning_rate, logs):
    message = (
        f"<phase:{phase_name}, epoch:{epoch:3d}, "
        f"phase_iter:{phase_step:8,d}, global_iter:{global_step:8,d}, "
        f"lr:{learning_rate:.3e}> "
    )
    for key, value in logs.items():
        if key.startswith("time_") or (key.startswith("window_") and key != "window_steps"):
            message += f"{key.replace('time_', '')}: {float(value):.4f}s "
        elif isinstance(value, numbers.Real):
            message += f"{key}: {float(value):.3e} "
        else:
            message += f"{key}: {value} "
    return message


def build_checkpoint_window_summary(window_logs):
    if not window_logs:
        return {}

    summary = {"window_steps": float(len(window_logs))}
    keys = sorted({
        key
        for logs in window_logs
        for key, value in logs.items()
        if key.startswith("time_") and isinstance(value, numbers.Real)
    })
    for key in keys:
        values = [
            float(logs[key])
            for logs in window_logs
            if isinstance(logs.get(key), numbers.Real)
        ]
        if values:
            metric = key.removeprefix("time_")
            summary[f"window_{metric}_mean"] = sum(values) / len(values)
            summary[f"window_{metric}_max"] = max(values)
    return summary


def verify_phase_checkpoint_paths(*, phase_name, checkpoint_g, checkpoint_e=None):
    if not checkpoint_g or not os.path.isfile(checkpoint_g):
        raise FileNotFoundError(
            f"{phase_name} G checkpoint is not ready: {checkpoint_g}. "
            "Phase handoff requires rank0 to finish saving before phase2 can load."
        )
    if checkpoint_e and not os.path.isfile(checkpoint_e):
        raise FileNotFoundError(
            f"{phase_name} E checkpoint is not ready: {checkpoint_e}. "
            "Phase handoff requires rank0 to finish saving before phase2 can load."
        )


def finalize_phase(*, model, phase_opt, phase_name, last_phase_step, last_global_step, shared_runtime):
    rank = phase_opt.get("rank", 0)
    if rank == 0:
        model.save(last_global_step)
        use_lora = (phase_opt.get("train") or {}).get("use_lora", False)
        if use_lora and hasattr(model, "save_merged"):
            model.save_merged(last_global_step)

    models_dir = (phase_opt.get("path") or {}).get("models", "")
    if models_dir:
        final_checkpoint_G = os.path.join(models_dir, f"{last_global_step}_G.pth")
        e_decay = (phase_opt.get("train") or {}).get("E_decay", 0)
        final_checkpoint_E = os.path.join(models_dir, f"{last_global_step}_E.pth") if e_decay else None
    else:
        final_checkpoint_G = None
        final_checkpoint_E = None

    if phase_opt.get("dist", False):
        barrier_safe()
    verify_phase_checkpoint_paths(
        phase_name=phase_name,
        checkpoint_g=final_checkpoint_G,
        checkpoint_e=final_checkpoint_E,
    )
    if phase_opt.get("dist", False):
        barrier_safe()

    return {
        "final_checkpoint_G": final_checkpoint_G,
        "final_checkpoint_E": final_checkpoint_E,
    }


def build_shared_runtime(opt, logger, tb_logger, seed):
    return {
        "logger": logger,
        "tb_logger": tb_logger,
        "seed": seed,
        "opt": opt,
    }


def close_shared_runtime(shared_runtime):
    tb_logger = shared_runtime.get("tb_logger")
    if tb_logger is not None:
        tb_logger.close()


def prepare_phase2_opt(phase2_opt, *, phase1_final_g, phase1_final_e):
    updated = copy.deepcopy(phase2_opt)
    updated.setdefault("path", {})
    updated.setdefault("train", {})
    updated["path"]["pretrained_netG"] = phase1_final_g
    updated["path"]["pretrained_netE"] = phase1_final_e
    updated["path"]["pretrained_optimizerG"] = None
    updated["train"]["G_optimizer_reuse"] = False
    return updated


def compute_phase2_resume_step(state):
    global_step_offset = int(state.get("global_step_offset", 0) or 0)
    last_global_step = int(state.get("last_successful_global_step", global_step_offset) or 0)
    return max(last_global_step - global_step_offset, 0)


def run_experiment(opt, logger, tb_logger, seed):
    phase1_opt, phase2_opt = resolve_two_run_phase_opts(opt)
    if phase1_opt is None or phase2_opt is None:
        raise ValueError("run_experiment requires train.two_run.enable=true")

    dump_resolved_two_run_opts(opt, phase1_opt, phase2_opt)
    shared_runtime = build_shared_runtime(opt, logger, tb_logger, seed)
    state_path = two_run_state_path(opt)
    state = load_two_run_state(state_path)
    shared_runtime["two_run_state_path"] = state_path
    if state is None:
        state = build_initial_two_run_state(
            phase1_total_iter=phase1_opt["train"]["total_iter"],
            phase2_total_iter=phase2_opt["train"]["total_iter"],
        )
        save_two_run_state(state_path, state)
    shared_runtime["two_run_state"] = state

    resume_phase = resolve_resume_phase(state)
    result = None
    try:
        if resume_phase in {"phase1_fresh", "phase1_resume"}:
            result = run_phase(
                phase_opt=phase1_opt,
                shared_runtime=shared_runtime,
                phase_name="phase1",
                global_step_offset=0,
                resume_state={"phase_step": 0 if resume_phase == "phase1_fresh" else state.get("last_successful_phase_step", 0)},
            )
            mark_phase1_completed(
                state,
                phase1_final_g=result.get("final_checkpoint_G"),
                phase1_final_e=result.get("final_checkpoint_E"),
            )
            save_two_run_state(state_path, state)

        prepared_phase2_opt = prepare_phase2_opt(
            phase2_opt,
            phase1_final_g=state["phase1_final_G"],
            phase1_final_e=state["phase1_final_E"],
        )
        phase2_resume_step = compute_phase2_resume_step(state) if resume_phase == "phase2_resume" else 0
        if resume_phase == "phase2_resume":
            state["last_successful_phase_step"] = phase2_resume_step
            state["last_successful_global_step"] = state["global_step_offset"] + phase2_resume_step
        mark_phase2_started(state)
        save_two_run_state(state_path, state)
        result = run_phase(
            phase_opt=prepared_phase2_opt,
            shared_runtime=shared_runtime,
            phase_name="phase2",
            global_step_offset=state["global_step_offset"],
            resume_state={"phase_step": phase2_resume_step},
        )
        return result
    finally:
        close_shared_runtime(shared_runtime)


def run_phase(phase_opt, shared_runtime, phase_name, global_step_offset, resume_state):
    logger = shared_runtime["logger"]
    tb_logger = shared_runtime["tb_logger"]
    seed = shared_runtime["seed"]
    runtime = build_phase_runtime(phase_opt, seed, logger, phase_name)
    model = runtime["model"]
    train_loader = runtime["train_loader"]
    train_sampler = runtime.get("train_sampler")
    test_loader = runtime.get("test_loader")
    phase_total_iter = phase_opt["train"]["total_iter"]
    phase_timing_enabled = bool(((phase_opt.get("train") or {}).get("timing") or {}).get("enable", False))
    last_phase_step = int((resume_state or {}).get("phase_step", 0))
    last_global_step = compute_global_step(global_step_offset, last_phase_step)

    epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    train_loader_iter = iter(train_loader)
    checkpoint_window_logs = []
    while not should_finish_phase(last_phase_step, phase_total_iter):
        iter_total_start = time.perf_counter() if phase_timing_enabled else None
        phase_step = last_phase_step + 1
        global_step = compute_global_step(global_step_offset, phase_step)
        model.timer.current_timings.clear()
        try:
            train_data = next_train_batch_with_timing(model, train_loader_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loader_iter = iter(train_loader)
            train_data = next_train_batch_with_timing(model, train_loader_iter)
        logs = execute_training_iteration(
            model=model,
            train_data=train_data,
            phase_step=phase_step,
            global_step=global_step,
        )
        if should_run_validation_checkpoint(phase_opt["train"].get("checkpoint_test"), phase_step):
            run_validation_checkpoint(
                model=model,
                phase_opt=phase_opt,
                test_loader=test_loader,
                tb_logger=tb_logger,
                logger=logger,
                phase_step=phase_step,
                global_step=global_step,
                epoch=epoch,
            )
        last_phase_step = phase_step
        last_global_step = global_step
        _state = shared_runtime.get("two_run_state")
        _state_path = shared_runtime.get("two_run_state_path")
        if _state is not None and _state_path is not None:
            state_persist_start = time.perf_counter() if phase_timing_enabled else None
            update_last_successful_step(_state, phase_step=phase_step, global_step=global_step)
            save_two_run_state(_state_path, _state)
            if phase_timing_enabled:
                logs["time_state_persist"] = time.perf_counter() - state_persist_start
        if phase_timing_enabled:
            logs["time_iter_total"] = time.perf_counter() - iter_total_start
        checkpoint_window_logs.append(dict(logs))
        if (
            logger is not None
            and phase_opt.get("rank", 0) == 0
            and phase_step % phase_opt["train"]["checkpoint_print"] == 0
        ):
            if phase_timing_enabled:
                logs.update(build_checkpoint_window_summary(checkpoint_window_logs))
            logger.info(
                format_training_log_message(
                    phase_name=phase_name,
                    epoch=epoch,
                    phase_step=phase_step,
                    global_step=global_step,
                    learning_rate=model.current_learning_rate(),
                    logs=logs,
                )
            )
            checkpoint_window_logs.clear()
        if tb_logger is not None and phase_step % phase_opt["train"]["checkpoint_print"] == 0:
            numeric_logs = {
                key: float(value)
                for key, value in logs.items()
                if isinstance(value, numbers.Real)
            }
            if numeric_logs:
                tb_logger.log_scalars(global_step, numeric_logs, tag_prefix="train")

    finalize = finalize_phase(
        model=model,
        phase_opt=phase_opt,
        phase_name=phase_name,
        last_phase_step=last_phase_step,
        last_global_step=last_global_step,
        shared_runtime=shared_runtime,
    )
    return {
        "model": model,
        "runtime": runtime,
        "last_phase_step": last_phase_step,
        "last_global_step": last_global_step,
        **finalize,
    }


def main():
    """
    主训练函数，接收命令行参数指定的配置文件路径
    """
    
    '''
    # ----------------------------------------
    # 步骤 1: 准备配置选项 (prepare opt)
    # ----------------------------------------
    '''
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='配置文件 JSON 文件路径')
    # 以下参数为向后兼容保留，但会被自动检测忽略
    parser.add_argument('--launcher', default='pytorch', help='任务启动器（已忽略，自动检测）')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程排名（已忽略，使用环境变量）')
    parser.add_argument('--dist', default=False, help='分布式模式（已忽略，自动检测）')
    
    args = parser.parse_args()  # 解析命令行参数

    # ----------------------------------------
    # 首先初始化分布式训练环境
    # ----------------------------------------
    # 在命令行解析之后、加载 JSON 配置之前立即初始化，以确保正确的设备设置
    # 这会检测环境变量（如 WORLD_SIZE, RANK）并设置分布式训练
    setup_distributed()
    
    # ----------------------------------------
    # 解析配置文件并自动检测分布式模式
    # ----------------------------------------
    # 从 JSON 文件加载所有训练配置（学习率、批次大小、模型参数等）
    opt = option.parse(args.opt, is_train=True)
    # opt['dist'] 已由 option.parse() 根据 WORLD_SIZE 环境变量自动设置
    # 获取当前进程的排名和总进程数
    opt['rank'], opt['world_size'] = get_dist_info()
    runtime_cpu_summary = apply_runtime_cpu_config(opt)
    
    # ----------------------------------------
    # 创建必要的目录（仅主进程 rank 0 执行）
    # ----------------------------------------
    # 在分布式训练中，只有主进程创建目录，避免多进程竞争
    if is_main_process():
        # 创建所有必要的目录（模型保存、日志、图像输出等），但不包括预训练模型路径
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # 更新配置选项：查找并加载检查点
    # ----------------------------------------
    # 查找最新的检查点文件，用于恢复训练或加载预训练模型
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    
    # 查找生成器网络 G 的最新检查点
    # init_iter_G: 检查点对应的迭代次数，init_path_G: 检查点文件路径
    # 如果 pretrained_netG 为 None/null，则从头开始训练，不自动查找检查点
    # 如果 pretrained_netG 为 "auto"，则自动查找检查点（用于恢复训练）
    # 如果 pretrained_netG 为其他字符串（指定了路径），则使用指定的路径，不自动查找检查点
    if opt['path']['pretrained_netG'] is None:
        init_iter_G = 0
        init_path_G = None
    elif opt['path']['pretrained_netG'] == "auto":
        # "auto" 表示自动查找检查点（恢复训练）
        init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',
                                                               pretrained_path=None)
    else:
        # 指定了路径，直接使用指定的路径
        init_iter_G = 0  # 无法从路径中提取迭代次数，设为 0
        init_path_G = opt['path']['pretrained_netG']
    
    # 查找编码器网络 E 的最新检查点
    # 如果 pretrained_netE 为 None/null，则不自动查找检查点
    # 如果 pretrained_netE 为 "auto"，则自动查找检查点（用于恢复训练）
    # 如果 pretrained_netE 为其他字符串（指定了路径），则使用指定的路径，不自动查找检查点
    if opt['path']['pretrained_netE'] is None:
        init_iter_E = 0
        init_path_E = None
    elif opt['path']['pretrained_netE'] == "auto":
        # "auto" 表示自动查找检查点（恢复训练）
        init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E',
                                                               pretrained_path=None)
    else:
        # 指定了路径，直接使用指定的路径
        init_iter_E = 0  # 无法从路径中提取迭代次数，设为 0
        init_path_E = opt['path']['pretrained_netE']
    
    # 更新配置中的预训练模型路径
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    
    # 查找优化器的最新检查点（只有当 G 或 E 有检查点时才查找优化器检查点）
    if init_iter_G > 0 or init_iter_E > 0:
        init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                                 net_type='optimizerG')
    else:
        init_iter_optimizerG = 0
        init_path_optimizerG = None
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    
    # 当前训练步数取所有检查点中最大的迭代次数（用于恢复训练）
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # 保存配置到 '../option.json' 文件
    # ----------------------------------------
    # 仅主进程保存，用于记录当前训练使用的配置
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # 将字典中缺失的键设置为 None
    # ----------------------------------------
    # 这样在访问不存在的键时不会报错，而是返回 None
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # 配置日志记录器
    # ----------------------------------------
    # 仅主进程初始化日志，避免多进程重复写入
    if opt['rank'] == 0:
        logger_name = 'train'
        # 设置日志文件路径
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'), opt=opt)
        logger = logging.getLogger(logger_name)
        # 记录完整的配置信息到日志
        logger.info(option.dict2str(opt))
        logger.info('[RUNTIME] cpu config applied: %s', runtime_cpu_summary)
        
        # 初始化 TensorBoard 和 WANDB 日志记录器
        # 用于可视化训练过程（损失曲线、学习率等）
        tb_logger = utils_logger.Logger(opt, logger)
    else:
        # 非主进程不初始化日志记录器
        logger = None
        tb_logger = None

    # ----------------------------------------
    # 设置随机种子（分布式训练时根据 rank 偏移）
    # ----------------------------------------
    # 获取配置中的手动种子，如果未设置则随机生成
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    
    # 为每个进程添加 rank 偏移，使不同进程有不同的随机状态
    # 这对于数据增强的多样性很重要，避免所有进程看到相同的数据
    seed_rank = seed + opt['rank']
    
    # 打印种子信息
    if is_main_process():
        print('Base random seed: {}'.format(seed))
        print('Rank {} using seed: {}'.format(opt['rank'], seed_rank))
    
    # 设置所有随机数生成器的种子，确保结果可复现
    random.seed(seed_rank)  # Python 随机数
    np.random.seed(seed_rank)  # NumPy 随机数
    torch.manual_seed(seed_rank)  # PyTorch CPU 随机数
    torch.cuda.manual_seed_all(seed_rank)  # PyTorch GPU 随机数（所有 GPU）

    # ----------------------------------------
    # Dispatch to two-run orchestrator if enabled
    # ----------------------------------------
    two_run_cfg = (opt.get("train") or {}).get("two_run") or {}
    if two_run_cfg.get("enable", False):
        run_experiment(opt, logger=logger, tb_logger=tb_logger, seed=seed)
        return

    '''
    # ----------------------------------------
    # 步骤 2: 创建数据加载器 (create dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) 创建数据集
    # 2) 为训练集和测试集创建数据加载器
    # ----------------------------------------
    # 遍历配置中的所有数据集（通常包括 'train' 和 'test'）
    test_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # 创建训练数据集
            if opt['rank'] == 0:
                logger.info('[DATASET] Creating training dataset...')
            mem_before_train = log_memory_stage(logger, 'Before creating train dataset', opt['rank'])

            train_dataset_opt_base = opt["datasets"]["train"]
            is_phase1 = compute_is_phase1(current_step, opt["train"].get("fix_iter", 0))
            bundle = build_train_loader_bundle(opt, train_dataset_opt_base, is_phase1, seed, logger)
            train_set = bundle["train_set"]
            train_sampler = bundle["train_sampler"]
            train_loader = bundle["train_loader"]
            active_train_dataset_opt = bundle["dataset_opt"]

            mem_after_train = log_memory_stage(logger, 'After creating train dataset', opt['rank'])
            if opt['rank'] == 0:
                logger.info('[DATASET] Train dataset created. Memory delta: {:.2f} GB'.format(mem_after_train - mem_before_train))
                logger.info(
                    "[TRAIN_PHASE] phase=%s batch_size=%d gt_size=%d",
                    "phase1" if is_phase1 else "phase2",
                    active_train_dataset_opt["dataloader_batch_size"],
                    active_train_dataset_opt["gt_size"],
                )

            # 计算训练迭代次数（向上取整）
            train_size = int(math.ceil(len(train_set) / active_train_dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))

        elif phase == 'test':
            # 创建测试/验证数据集
            if opt['rank'] == 0:
                logger.info('[DATASET] Creating test/validation dataset...')
            mem_before_test = log_memory_stage(logger, 'Before creating test dataset', opt['rank'])

            test_set = define_Dataset(dataset_opt)

            mem_after_test = log_memory_stage(logger, 'After creating test dataset', opt['rank'])
            if opt['rank'] == 0:
                logger.info('[DATASET] Test dataset created. Memory delta: {:.2f} GB'.format(mem_after_test - mem_before_test))
                logger.info('[DATASET] Test dataset does not use in-memory caching (cache_data=False by default)')
            # 允许通过配置指定 DataLoader 的各项参数
            test_batch_size = dataset_opt.get('dataloader_batch_size', 1)
            test_num_workers = dataset_opt.get('dataloader_num_workers', 1)
            test_shuffle = dataset_opt.get('dataloader_shuffle', False)

            if opt['dist']:
                # 分布式验证 / 测试
                # 确保DistributedSampler使用正确的world_size和rank
                if dist.is_initialized():
                    test_sampler = DistributedSampler(
                        test_set, 
                        num_replicas=dist.get_world_size(),
                        rank=dist.get_rank(),
                        shuffle=test_shuffle,
                        drop_last=False, 
                        seed=seed
                    )
                else:
                    test_sampler = DistributedSampler(test_set, shuffle=test_shuffle,
                                                      drop_last=False, seed=seed)
                per_gpu_batch_size = max(1, test_batch_size // opt['num_gpu'])
                per_gpu_num_workers = max(1, test_num_workers // opt['num_gpu'])
                # DistributedSampler会自动将数据集均匀分配给各个rank
                # 当数据集大小不能被world_size整除时，会尽可能均匀分配
                # 例如：9个样本，3个GPU -> 每个GPU分配3个样本
                #      10个样本，3个GPU -> rank0:4个, rank1:3个, rank2:3个
                # drop_last=False确保所有数据都会被处理，即使分配不完全均匀
                test_loader_kwargs = dict(
                    batch_size=per_gpu_batch_size,
                    shuffle=False,
                    num_workers=per_gpu_num_workers,
                    drop_last=False,
                    pin_memory=True,
                    sampler=test_sampler,
                )
                if per_gpu_num_workers > 0:
                    test_loader_kwargs['persistent_workers'] = dataset_opt.get('dataloader_persistent_workers', False)
                    test_loader_kwargs['prefetch_factor'] = dataset_opt.get('dataloader_prefetch_factor', 2)
                    test_loader_kwargs['multiprocessing_context'] = 'spawn'
                test_loader = DataLoader(test_set, **test_loader_kwargs)
            else:
                # 单卡验证 / 测试
                test_loader_kwargs = dict(
                    batch_size=test_batch_size,
                    shuffle=test_shuffle,
                    num_workers=test_num_workers,
                    drop_last=False,
                    pin_memory=True,
                )
                if test_num_workers > 0:
                    test_loader_kwargs['persistent_workers'] = dataset_opt.get('dataloader_persistent_workers', False)
                    test_loader_kwargs['prefetch_factor'] = dataset_opt.get('dataloader_prefetch_factor', 2)
                    test_loader_kwargs['multiprocessing_context'] = 'spawn'
                test_loader = DataLoader(test_set, **test_loader_kwargs)

            requested_dist_patch_val = bool(opt.get('val', {}).get('distributed_patch_testing', False))
            dist_world_size = dist.get_world_size() if opt['dist'] and dist.is_initialized() else opt.get('world_size', 1)
            active_dist_patch_val = requested_dist_patch_val and opt['dist'] and dist_world_size > 1 and len(test_set) == 1
            opt.setdefault('val', {})['distributed_patch_testing_active'] = active_dist_patch_val
            if requested_dist_patch_val and opt['rank'] == 0:
                if active_dist_patch_val:
                    logger.info(
                        '[VALIDATION] Distributed patch testing active for single-sample validation '
                        f'(world_size={dist_world_size}).'
                    )
                else:
                    logger.info(
                        '[VALIDATION] Distributed patch testing requested but inactive '
                        f'(dist={opt["dist"]}, world_size={dist_world_size}, test_set_size={len(test_set)}).'
                    )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # 步骤 3: 初始化模型 (initialize model)
    # ----------------------------------------
    '''

    # 根据配置创建模型（VRT 或 RVRT）
    model = define_Model(opt)
    # 初始化训练相关组件（优化器、损失函数、学习率调度器等）
    model.init_train()
    if opt['rank'] == 0:
        # 关闭初始化阶段的网络结构/参数明细打印，避免训练日志开头过长。
        # 记录网络结构和参数信息。
        # logger.info(model.info_network())  # 网络架构信息
        # logger.info(model.info_params())  # 参数/权重统计信息
        pass

    '''
    # ----------------------------------------
    # 步骤 4: 主训练循环 (main training)
    # ----------------------------------------
    '''

    profiler_cfg = TrainProfilerConfig.from_opt(
        opt.get("train", {}),
        experiment_dir=opt["path"]["task"],
        rank=opt["rank"],
    )
    train_profiler = TrainProfiler(profiler_cfg, logger=logger if opt["rank"] == 0 else None)
    train_profiler.maybe_start()

    try:
        fix_iter = opt["train"].get("fix_iter", 0)
        last_is_phase1 = is_phase1

        # 开始训练循环（最多运行 1000000 个 epoch，实际由 total_iter 控制）
        for epoch in range(1000000):  # 持续运行直到达到总迭代次数
            # 为 DistributedSampler 设置 epoch，确保每个 epoch 数据打乱顺序不同
            if opt['dist']:
                train_sampler.set_epoch(epoch)
        
            # 遍历训练数据
            train_loader_iter = iter(train_loader)
            while True:
                next_step = current_step + 1
                is_phase1_now = compute_is_phase1(next_step, fix_iter)
                if is_phase1_now != last_is_phase1:
                    bundle = build_train_loader_bundle(opt, train_dataset_opt_base, is_phase1_now, seed, logger)
                    train_set = bundle["train_set"]
                    train_sampler = bundle["train_sampler"]
                    train_loader = bundle["train_loader"]
                    active_train_dataset_opt = bundle["dataset_opt"]
                    if opt['dist']:
                        train_sampler.set_epoch(epoch)
                    train_loader_iter = iter(train_loader)

                    if opt["rank"] == 0:
                        logger.info(
                            "[TRAIN_PHASE] switch=%s batch_size=%d gt_size=%d (rebuild train loader)",
                            "phase1" if is_phase1_now else "phase2",
                            active_train_dataset_opt["dataloader_batch_size"],
                            active_train_dataset_opt["gt_size"],
                        )
                    last_is_phase1 = is_phase1_now

                model.timer.current_timings.clear()
                try:
                    with model.timer.timer('batch_wait'):
                        train_data = next(train_loader_iter)
                except StopIteration:
                    break

                current_step = next_step  # 更新当前训练步数

                # -------------------------------
                # 1) 更新学习率
                # -------------------------------
                # 根据当前步数调整学习率（可能使用学习率调度器）
                model.update_learning_rate(current_step)

                # -------------------------------
                # 2) 输入数据对（低质量图像和高质量图像）
                # -------------------------------
                # 将训练数据（低质量输入和高质量标签）送入模型
                model.feed_data(train_data, current_step=current_step)

                # -------------------------------
                # 3) 优化模型参数
                # -------------------------------
                # 执行前向传播、计算损失、反向传播和参数更新
                model.optimize_parameters(current_step)
                train_profiler.step(current_step)

                # -------------------------------
                # 4) 记录训练信息
                # -------------------------------
                # 每隔一定步数打印训练信息（损失值、学习率等）
                if current_step % opt['train']['checkpoint_print'] == 0:
                    logs = build_timing_summary(
                        model.current_log(),
                        dist_enabled=opt.get('dist', False),
                        device=model.device,
                    )
                if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                    # 构建日志消息：包含 epoch、迭代次数、学习率
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                              model.current_learning_rate())
                    seconds_per_iter = logs.get('time_total', logs.get('time_iter'))
                    if isinstance(seconds_per_iter, numbers.Real):
                        message += 'eta: {:s} '.format(
                            compute_training_eta(current_step, opt['train']['total_iter'], seconds_per_iter)
                        )
                    # 将损失信息合并到消息中
                    for k, v in logs.items():  # 合并日志信息到消息
                        if k.startswith('time_'):
                            # 耗时信息使用不同的格式
                            message += '{:s}: {:.4f}s '.format(k.replace('time_', ''), v)
                        elif isinstance(v, numbers.Real):
                            message += '{:s}: {:.3e} '.format(k, float(v))
                        else:
                            message += '{:s}: {} '.format(k, v)
                    logger.info(message)  # 写入日志文件
                
                    # 记录到 TensorBoard 和 WANDB（用于可视化）
                    if tb_logger is not None:
                        # 分离训练指标和耗时指标，分别记录到不同命名空间
                        train_logs = {
                            k: float(v)
                            for k, v in logs.items()
                            if not k.startswith('time_') and isinstance(v, numbers.Real)
                        }
                        train_logs['learning_rate'] = model.current_learning_rate()
                        # 耗时指标去掉time_前缀，记录到time命名空间
                        time_logs = {k.replace('time_', ''): v for k, v in logs.items() if k.startswith('time_')}
                    
                        if train_logs:
                            tb_logger.log_scalars(current_step, train_logs, tag_prefix='train')
                        if time_logs:
                            tb_logger.log_scalars(current_step, time_logs, tag_prefix='time')

                # -------------------------------
                # 5) 保存模型检查点
                # -------------------------------
                # 每隔一定步数保存模型（包括网络权重和优化器状态）
                if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                    logger.info('Saving the model.')
                    model.save(current_step)  # 保存当前步数的模型
                # 在分布式训练中，等待 rank 0 完成保存，避免进程间状态不一致
                if current_step % opt['train']['checkpoint_save'] == 0 and opt['dist']:
                    barrier_safe()  # 同步所有进程


                # 特殊处理：当使用静态计算图时，在改变计算图之前提前保存模型
                # 这是因为在分布式训练中使用 use_checkpoint=True 时存在 bug
                if opt['use_static_graph'] and (current_step == opt['train']['fix_iter'] - 1):
                    current_step += 1
                    model.update_learning_rate(current_step)
                    if opt['rank'] == 0:
                        model.save(current_step)  # 提前保存模型
                    # 等待 rank 0 完成保存
                    if opt['dist']:
                        barrier_safe()
                    current_step -= 1  # 恢复步数
                    if opt['rank'] == 0:
                        logger.info('Saving models ahead of time when changing the computation graph with use_static_graph=True'
                                    ' (we need it due to a bug with use_checkpoint=True in distributed training). The training '
                                    'will be terminated by PyTorch in the next iteration. Just resume training with the same '
                                    '.json config file.')

                # -------------------------------
                # 6) 模型测试和评估
                # -------------------------------
                # 每隔一定步数在测试集上评估模型性能
                _ckpt_test = opt['train']['checkpoint_test']
                if isinstance(_ckpt_test, list):
                    _do_test = current_step in _ckpt_test
                else:
                    _do_test = current_step % _ckpt_test == 0
                if _do_test:

                    if opt['rank'] == 0:
                        logger.info('[VALIDATION] Starting model validation/test phase...')
                    mem_before_validation = log_memory_stage(logger, 'Before validation memory cleanup', opt['rank'])


                    is_master_process = opt['rank'] == 0
                    if opt['dist']:
                        barrier_safe()

                    # 初始化测试结果字典，用于存储所有文件夹的指标
                    test_results = OrderedDict()
                    test_results['psnr'] = []  # 峰值信噪比（RGB 通道）
                    test_results['ssim'] = []  # 结构相似性指数（RGB 通道）
                    test_results['psnr_y'] = []  # 峰值信噪比（Y 通道，亮度）
                    test_results['ssim_y'] = []  # 结构相似性指数（Y 通道，亮度）

                    # 初始化 gt 变量，避免在循环外部引用时出现 UnboundLocalError
                    gt = None

                    # 遍历测试数据
                    for idx, test_data in enumerate(test_loader):
                        if opt['rank'] == 0:
                            maybe_dump_full_frame_fusion_debug_from_batch(model, test_data, current_step, idx)
                        if idx < 2:
                            log_validation_probe(logger, f'before feed_data batch={idx}', opt['rank'])
                            if opt['rank'] == 0:
                                batch_summary = []
                                for key, value in test_data.items():
                                    if hasattr(value, 'shape'):
                                        batch_summary.append(f'{key}:{tuple(value.shape)}')
                                    elif isinstance(value, (list, tuple)):
                                        batch_summary.append(f'{key}:len={len(value)}')
                                    else:
                                        batch_summary.append(f'{key}:{type(value).__name__}')
                                logger.info(f'[VAL_BATCH] idx={idx} contents: {", ".join(batch_summary)}')
                        # 将测试数据送入模型
                        model.feed_data(test_data)
                        if idx < 2:
                            log_validation_probe(logger, f'after feed_data batch={idx}', opt['rank'])
                        # 执行测试（前向传播，不更新梯度）
                        if idx < 2 and opt['rank'] == 0:
                            logger.info(f'[VAL_BATCH] idx={idx} entering model.test()')
                        model.test()
                        if idx < 2:
                            log_validation_probe(logger, f'after model.test batch={idx}', opt['rank'])
                            if opt['rank'] == 0:
                                logger.info(f'[VAL_BATCH] idx={idx} model.test() completed')

                        if not should_score_validation_batch(opt):
                            continue

                        # 获取模型输出和真实标签
                        visuals = model.current_visuals()
                        output = visuals['E']  # E: 估计/输出图像 (Estimated)
                        gt = visuals['H'] if 'H' in visuals else None  # H: 高质量真实图像 (High-quality)
                        folder = test_data['folder']  # 测试序列的文件夹名称
                        folder_name = folder[0] if isinstance(folder, (list, tuple)) else folder
                        total_test_batches = len(test_loader)

                        # 初始化当前测试序列的结果字典
                        test_results_folder = OrderedDict()
                        test_results_folder['psnr'] = []
                        test_results_folder['ssim'] = []
                        test_results_folder['psnr_y'] = []
                        test_results_folder['ssim_y'] = []

                        # 处理批次中的每一张图像
                        lq_paths = test_data.get('lq_path')
                        batch_clip_count = output.shape[0]
                        for i in range(batch_clip_count):
                            clip_name = f'clip_{i:03d}'
                            if lq_paths is not None:
                                clip_source = None
                                try:
                                    clip_source = lq_paths[i]
                                except Exception:
                                    clip_source = None
                                if isinstance(clip_source, (list, tuple)) and clip_source:
                                    clip_source = clip_source[0]
                                if isinstance(clip_source, str):
                                    clip_name = os.path.splitext(os.path.basename(clip_source))[0]
                            # -----------------------
                            # 保存估计的图像 E
                            # -----------------------
                            # 将输出张量转换为 numpy 数组，并限制在 [0, 1] 范围内
                            img = output[i, ...].clamp_(0, 1).numpy()
                            if img.ndim == 3:
                                # 从 CHW-RGB 格式转换为 HWC-BGR 格式（OpenCV 使用 BGR）
                                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
                            # 将浮点数 [0, 1] 转换为 uint8 [0, 255]
                            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8

                            # 如果配置要求保存图像，则保存到磁盘
                            if opt['val']['save_img']:
                                save_dir = opt['path']['images']
                                util.mkdir(save_dir)
                                # 创建文件夹并保存图像
                                os.makedirs(f'{save_dir}/{folder_name}', exist_ok=True)
                                cv2.imwrite(f'{save_dir}/{folder_name}/{clip_name}_{current_step:d}.png', img)

                            # -----------------------
                            # 计算 PSNR 和 SSIM
                            # -----------------------
                            # 只有在有真实标签时才计算指标
                            if gt is not None:
                                # 处理真实标签图像
                                img_gt = gt[i, ...].clamp_(0, 1).numpy()
                                if img_gt.ndim == 3:
                                    # 从 CHW-RGB 格式转换为 HWC-BGR 格式
                                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
                                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                                img_gt = np.squeeze(img_gt)  # 移除单维度

                                # 计算 RGB 通道的 PSNR 和 SSIM
                                clip_psnr = util.calculate_psnr(img, img_gt, border=0)
                                clip_ssim = util.calculate_ssim(img, img_gt, border=0)
                                test_results_folder['psnr'].append(clip_psnr)
                                test_results_folder['ssim'].append(clip_ssim)

                                # 如果是 RGB 图像，计算 Y 通道（亮度）的指标
                                if img_gt.ndim == 3:  # RGB image
                                    # 转换为 YCbCr 颜色空间，只取 Y 通道（亮度）
                                    img_y = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                                    img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                                    # 计算 Y 通道的 PSNR 和 SSIM
                                    clip_psnr_y = util.calculate_psnr(img_y, img_gt_y, border=0)
                                    clip_ssim_y = util.calculate_ssim(img_y, img_gt_y, border=0)
                                else:
                                    # 灰度图像，Y 通道指标等于 RGB 指标
                                    clip_psnr_y = clip_psnr
                                    clip_ssim_y = clip_ssim
                                test_results_folder['psnr_y'].append(clip_psnr_y)
                                test_results_folder['ssim_y'].append(clip_ssim_y)

                        # 如果有计算的指标，记录并保存结果
                        if len(test_results_folder['psnr']) > 0:
                            # 计算当前测试序列的平均指标
                            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
                            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])

                            test_results['psnr'].append(psnr)
                            test_results['ssim'].append(ssim)
                            test_results['psnr_y'].append(psnr_y)
                            test_results['ssim_y'].append(ssim_y)

                            # 打印每个文件夹的结果（保持原有的打印行为）
                            print('[Rank {}] Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                                  format(opt['rank'], folder_name, len(test_results['psnr']), len(test_loader), psnr, ssim, psnr_y, ssim_y))
                        else:
                            # 没有真实标签时，只记录测试序列名称
                            pass


                    # 计算全局平均值和最大值
                    local_psnr_sum = sum(test_results['psnr'])
                    local_ssim_sum = sum(test_results['ssim'])
                    local_psnr_y_sum = sum(test_results['psnr_y'])
                    local_ssim_y_sum = sum(test_results['ssim_y'])

                    local_psnr_count = len(test_results['psnr'])
                    local_ssim_count = len(test_results['ssim'])
                    local_psnr_y_count = len(test_results['psnr_y'])
                    local_ssim_y_count = len(test_results['ssim_y'])

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    metrics_tensor = torch.tensor(
                        [local_psnr_sum, local_ssim_sum, local_psnr_y_sum, local_ssim_y_sum,
                         local_psnr_count, local_ssim_count, local_psnr_y_count, local_ssim_y_count],
                        dtype=torch.float64, device=device)

                    world_size = opt.get('world_size', 1)

                    if opt['dist'] and dist.is_initialized():
                        # Gather per-rank folder metrics so rank 0 can print each rank's results and compute global stats
                        gathered = [torch.zeros_like(metrics_tensor) for _ in range(world_size)]
                        dist.all_gather(gathered, metrics_tensor)

                        if is_master_process:
                            # Print per-rank average results and collect all folder metrics for global max
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
                                    logger.info('[Rank {}] Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                                                format(r, ave_psnr_r, ave_ssim_r, ave_psnr_y_r, ave_ssim_y_r))

                                    # Collect all folder metrics for global max calculation
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

                                logger.info('<epoch:{:3d}, iter:{:8,d} Max PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                                    epoch, current_step, max_psnr, max_ssim, max_psnr_y, max_ssim_y))

                                logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                                    epoch, current_step, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))

                                # 将测试指标记录到 TensorBoard 和 WANDB (使用PSNR最大的clip的完整指标)
                                if tb_logger is not None:
                                    test_metrics = {
                                        'psnr': max_psnr,
                                        'ssim': max_ssim,
                                        'psnr_y': max_psnr_y,
                                        'ssim_y': max_ssim_y
                                    }
                                    tb_logger.log_scalars(current_step, test_metrics, tag_prefix='test')
                    else:
                        # Non-distributed: just compute local max and average
                        if is_master_process:
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

                                logger.info('<epoch:{:3d}, iter:{:8,d} Max PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                                    epoch, current_step, max_psnr, max_ssim, max_psnr_y, max_ssim_y))

                                logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                            'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                                    epoch, current_step, avg_psnr, avg_ssim, avg_psnr_y, avg_ssim_y))

                                # 将测试指标记录到 TensorBoard 和 WANDB (使用PSNR最大的clip的完整指标)
                                if tb_logger is not None:
                                    test_metrics = {
                                        'psnr': max_psnr,
                                        'ssim': max_ssim,
                                        'psnr_y': max_psnr_y,
                                        'ssim_y': max_ssim_y
                                    }
                                    tb_logger.log_scalars(current_step, test_metrics, tag_prefix='test')

                    mem_after_validation = log_memory_stage(logger, 'After validation completion', opt['rank'])
                    if opt['rank'] == 0:
                        logger.info('[VALIDATION] Validation phase completed. Memory usage: {:.2f} GB'.format(mem_after_validation))

                    if opt['dist']:
                        barrier_safe()

                # 检查是否达到总迭代次数，如果达到则结束训练
                if current_step > opt['train']['total_iter']:
                    if opt['rank'] == 0:
                        logger.info('Finish training.')
                        model.save(current_step)  # 保存最终模型
                        if hasattr(model, 'save_merged'):
                            model.save_merged(current_step)
                        if tb_logger is not None:
                            tb_logger.close()  # 关闭日志记录器
                    sys.exit()  # 退出程序
    finally:
        train_profiler.close()

# 主程序入口
if __name__ == '__main__':
    main()
