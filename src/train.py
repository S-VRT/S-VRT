import argparse
import contextlib
import gc
import math
import os
import sys
import time
import random
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import yaml

import datetime
import shutil
from src.utils.path_manager import create_experiment_dir, get_config_from_exp_dir, get_checkpoint_path
import re

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Memory management configuration
MEMORY_CEILING_GB = 200.0  # Maximum allowed system memory usage (80% of 250GB)
MEMORY_WARNING_GB = 180.0  # Warning threshold (72% of 250GB)
MEMORY_CHECK_INTERVAL = 10  # Check every N batches
CACHE_CLEAR_INTERVAL = 200  # Clear dataset cache every N steps (reduced from 500 to prevent accumulation)

# Set up Python paths for imports (CRITICAL for both project and VRT imports)
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
VRT_ROOT = REPO_ROOT / "third_party" / "VRT"

# Add project root to sys.path for 'src' imports
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add VRT to sys.path for VRT model imports
if str(VRT_ROOT) not in sys.path:
    sys.path.insert(0, str(VRT_ROOT))

from models.network_vrt import VRT  # type: ignore

from src.data.datasets.spike_deblur_dataset import SpikeDeblurDataset
from src.data.collate_fns import safe_spike_deblur_collate
from src.data.preprocessing import get_preprocessor
from src.losses import CharbonnierLoss, VGGPerceptualLoss
from src.models.integrate_vrt import VRTWithSpike
from src.utils.timing_logger import TimingLogger, set_global_timing_logger
from src.loggers import WandBLogger


def setup_logger(log_file: Path, rank: int = 0) -> logging.Logger:
    """
    设置logger，将日志同时输出到文件和控制台
    
    Args:
        log_file: 日志文件路径
        rank: 进程rank，只有rank 0会写日志
    
    Returns:
        配置好的logger对象
    """
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    if rank == 0:
        # 文件handler
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger




def parse_args():
    parser = argparse.ArgumentParser(description="Train VRT with Spike")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory for resume")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--profile_steps", type=int, default=100, help="Number of steps to profile")
    return parser.parse_args()


def format_seconds(seconds: float) -> str:
    """Format seconds into H:MM:SS for progress display."""
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def check_and_manage_memory(step: int, rank: int = 0, dataset = None) -> Tuple[bool, str]:
    """
    Check system memory and trigger cleanup if necessary.
    
    Args:
        step: Current training step
        rank: Process rank
        dataset: Dataset object (for cache clearing)
    
    Returns:
        Tuple of (needs_cleanup, message)
    """
    if not HAS_PSUTIL:
        return False, ""
    
    # Get memory info
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    
    # Check if we're approaching the ceiling
    if used_gb >= MEMORY_CEILING_GB:
        # Critical: Force aggressive cleanup
        msg = f"🚨 [Memory Ceiling] {used_gb:.1f}GB/{MEMORY_CEILING_GB:.1f}GB - FORCING CLEANUP"
        if rank == 0:
            print(f"\n{msg}", flush=True)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        
        # Clear dataset cache if available
        if dataset is not None and hasattr(dataset, 'clear_cache'):
            dataset.clear_cache()
            if rank == 0:
                print(f"    ✓ Cleared dataset cache", flush=True)
        
        # Force garbage collection
        gc.collect()
        
        # Wait a moment for memory to be released
        time.sleep(0.5)
        
        # Check new memory
        mem_after = psutil.virtual_memory()
        freed_gb = (mem.used - mem_after.used) / (1024**3)
        if rank == 0:
            print(f"    ✓ Freed {freed_gb:.1f}GB, now at {mem_after.used/(1024**3):.1f}GB", flush=True)
        
        return True, msg
    
    elif used_gb >= MEMORY_WARNING_GB:
        # Warning: Gentle cleanup
        msg = f"⚠️  [Memory Warning] {used_gb:.1f}GB/{MEMORY_CEILING_GB:.1f}GB"
        if rank == 0 and step % 50 == 0:
            print(f"\n{msg}", flush=True)
        
        # Gentle cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True, msg
    
    return False, ""

def prune_step_checkpoints(ckpt_dir: Path, keep_last_n: int):
    if keep_last_n <= 0:
        return  # A value of 0 or less means keep all

    # Find, sort, and filter step-based checkpoints
    step_checkpoints = sorted(
        [p for p in ckpt_dir.glob("step_*.pth")],
        key=lambda p: int(re.search(r"step_(\d+).pth", str(p)).group(1))
    )

    # Remove the oldest checkpoints
    if len(step_checkpoints) > keep_last_n:
        for p in step_checkpoints[:-keep_last_n]:
            p.unlink()


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    在checkpoints目录中查找最新的checkpoint文件
    
    Args:
        checkpoint_dir: checkpoints目录路径
    
    Returns:
        最新checkpoint的路径，如果没有找到则返回None
    """
    if not checkpoint_dir.exists():
        return None
    
    # 查找所有.pth文件
    ckpt_files = list(checkpoint_dir.glob("*.pth"))
    if not ckpt_files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_ckpt = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    return latest_ckpt


def get_optimal_num_workers(config: Dict, world_size: int) -> Dict[str, int]:
    """
    计算最优的DataLoader worker数量
    
    Args:
        config: YAML配置字典
        world_size: GPU数量
    
    Returns:
        包含train和val的num_workers字典
    """
    dataloader_cfg = config.get("DATALOADER", {})
    
    # 获取CPU核心数
    try:
        cpu_count = os.cpu_count() or 8
    except:
        cpu_count = 8
    
    # 从配置获取总worker数（支持auto自动分配）
    total_workers_cfg = dataloader_cfg.get("TOTAL_WORKERS", "auto")
    
    if total_workers_cfg == "auto":
        # 自动模式：使用80%的CPU核心用于数据加载，保留20%给系统和训练
        # 避免过度订阅导致上下文切换开销
        total_workers = max(1, int(cpu_count * 0.8))
    elif isinstance(total_workers_cfg, str) and total_workers_cfg.startswith("cpu*"):
        # 支持 "cpu*0.8" 这样的配置
        factor = float(total_workers_cfg.replace("cpu*", ""))
        total_workers = max(1, int(cpu_count * factor))
    else:
        # 直接指定worker数
        total_workers = int(total_workers_cfg)
    
    # 计算每GPU的worker数
    per_gpu_workers = max(1, total_workers // max(1, world_size))
    
    # 训练和验证的worker分配
    train_workers_cfg = dataloader_cfg.get("TRAIN_WORKERS")
    val_workers_cfg = dataloader_cfg.get("VAL_WORKERS")
    
    if train_workers_cfg is None:
        # 如果没有单独配置，使用per_gpu计算值
        train_workers = per_gpu_workers
    elif train_workers_cfg == "auto":
        train_workers = per_gpu_workers
    else:
        train_workers = int(train_workers_cfg)
    
    if val_workers_cfg is None:
        # 验证默认使用训练worker数的一半
        val_workers = max(1, train_workers // 2)
    elif val_workers_cfg == "auto":
        val_workers = max(1, per_gpu_workers // 2)
    else:
        val_workers = int(val_workers_cfg)
    
    return {
        'train': train_workers,
        'val': val_workers,
        'total': total_workers,
        'cpu_count': cpu_count,
    }


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable cudnn benchmark for better performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def log_memory_usage(rank: int = 0, prefix: str = "") -> None:
    """Log current memory usage (CPU and GPU)."""
    if not HAS_PSUTIL:
        return
    
    if rank != 0:  # Only log on rank 0
        return
    
    # CPU memory
    process = psutil.Process()
    cpu_mem_mb = process.memory_info().rss / (1024**2)
    cpu_percent = process.memory_percent()
    
    # System memory
    sys_mem = psutil.virtual_memory()
    sys_mem_total_gb = sys_mem.total / (1024**3)
    sys_mem_used_gb = sys_mem.used / (1024**3)
    sys_mem_percent = sys_mem.percent
    
    msg = f"[{prefix}] Memory - Process: {cpu_mem_mb:.1f}MB ({cpu_percent:.1f}%), "
    msg += f"System: {sys_mem_used_gb:.1f}/{sys_mem_total_gb:.1f}GB ({sys_mem_percent:.1f}%)"
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_mem_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        gpu_mem_reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        msg += f", GPU: {gpu_mem_allocated_mb:.1f}MB allocated, {gpu_mem_reserved_mb:.1f}MB reserved"
    
    print(msg)
    
    # Warning if system memory is critically high
    if sys_mem_percent > 90:
        print(f"⚠️  WARNING: System memory usage is critically high ({sys_mem_percent:.1f}%)!")
        print(f"   Consider reducing CACHE_SIZE_GB or NUM_WORKERS in config.")


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-10) -> float:
    # pred, gt: (B,T,3,H,W) in [0,1]
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    # x: (B,T,3,H,W) in [0,1]
    r = x[:, :, 0:1]
    g = x[:, :, 1:2]
    b = x[:, :, 2:3]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0
    return y


def compute_ssim(y_pred: torch.Tensor, y_gt: torch.Tensor, window_size: int = 11) -> float:
    """
    Compute SSIM on Y channel tensors in [0,1].
    y_pred, y_gt: (B,T,1,H,W) or (B,T,3,H,W) with caller providing Y.
    Returns mean SSIM over batch and time.
    """
    import torch.nn.functional as F
    if y_pred.shape[2] != 1:
        raise ValueError("compute_ssim expects single-channel inputs (Y)")
    # Merge B and T
    b, t, c, h, w = y_pred.shape
    x = y_pred.reshape(b * t, 1, h, w)
    y = y_gt.reshape(b * t, 1, h, w)
    # Gaussian window approximation with uniform kernel for simplicity
    pad = window_size // 2
    weight = torch.ones((1, 1, window_size, window_size), device=x.device) / (window_size * window_size)
    mu_x = F.conv2d(x, weight, padding=pad)
    mu_y = F.conv2d(y, weight, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, weight, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, weight, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, weight, padding=pad) - mu_xy
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return float(ssim_map.mean().item())


def main() -> None:
    args = parse_args()
    
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    # First load the base config to check for possible EXP_DIR
    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)

    # Determine exp_dir
    exp_dir = None
    is_resume = False

    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        # Check if this is actually a resume by looking for config snapshot
        # (launch_train.sh may create exp_dir for new runs too)
        config_snapshot_dir = exp_dir / "configs"
        if config_snapshot_dir.exists() and list(config_snapshot_dir.glob("*.yaml")):
            is_resume = True
        else:
            is_resume = False
    elif cfg.get("TRAIN", {}).get("EXP_DIR"):
        exp_dir = Path(cfg["TRAIN"]["EXP_DIR"])
        # Check if this is actually a resume
        config_snapshot_dir = exp_dir / "configs"
        if config_snapshot_dir.exists() and list(config_snapshot_dir.glob("*.yaml")):
            is_resume = True
        else:
            is_resume = False
    else:
        # Create new exp_dir
        config_name = Path(args.config).stem
        run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(cfg.get("LOG", {}).get("SAVE_DIR", "outputs")) / config_name
        exp_dir = base_dir / run_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train.py] Created new experiment directory: {exp_dir}")

    # Handle resume if applicable
    if is_resume:
        if not exp_dir.exists():
            raise ValueError(f"Specified exp_dir {exp_dir} does not exist. Cannot resume.")
        # Load config from exp_dir snapshot (ensures reproducibility)
        config_path = get_config_from_exp_dir(exp_dir)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: Dict = yaml.safe_load(f)  # Override with exp_dir config
        print(f"[train.py] Resuming from existing exp_dir: {exp_dir}")
        print(f"[train.py] Using config snapshot: {config_path}")
    else:
        # Save config snapshot for new run
        (exp_dir / "configs").mkdir(parents=True, exist_ok=True)
        config_snapshot_path = exp_dir / "configs" / Path(args.config).name
        shutil.copy2(args.config, config_snapshot_path)
        print(f"[train.py] Saved config snapshot: {config_snapshot_path}")

    # Create subdirectories (for both new and resume)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "visuals").mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # Multi-GPU setup
    if args.device == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        num_gpus = torch.cuda.device_count()
        if is_main_process:
            print(f"[train] Found {num_gpus} GPUs available")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")
            if world_size > 1:
                print(f"[train] Using DistributedDataParallel with {world_size} GPUs")
    else:
        device = torch.device(args.device)
        num_gpus = 0
    
    # Set environment variable for collate_fn to use
    # os.environ["DEBLUR_SAVE_DIR"] = str(exp_dir) # Assuming not needed, or set to str(exp_dir)

    # Seed & TensorBoard
    set_seed(int(cfg.get("SEED", 123)))
    writer = SummaryWriter(log_dir=str(exp_dir / "logs" / "tb")) if cfg["LOG"].get("TENSORBOARD", True) else None

    wandb_cfg = cfg["LOG"].get("WANDB", {})
    wandb_logger = WandBLogger(
        cfg=cfg,
        exp_dir=exp_dir,
        enable=bool(wandb_cfg.get("ENABLE", False)),
        project=wandb_cfg.get("PROJECT", "deblur"),
        entity=wandb_cfg.get("ENTITY"),
        run_name=wandb_cfg.get("RUN_NAME"),
        job_type=wandb_cfg.get("JOB_TYPE"),
        tags=wandb_cfg.get("TAGS"),
        resume=wandb_cfg.get("RESUME", "allow"),
        watch=wandb_cfg.get("WATCH"),
        log_checkpoints=bool(wandb_cfg.get("LOG_CHECKPOINTS", True)),
        log_images=bool(wandb_cfg.get("LOG_IMAGES", False)),
    )
    progress_every = int(cfg["LOG"].get("PROGRESS_EVERY_STEPS", 20))
    
    # Setup timing logger
    enable_timing_log = cfg["LOG"].get("ENABLE_TIMING_LOG", True)
    timing_logger = None
    if enable_timing_log and is_main_process:
        timing_logger = TimingLogger(
            log_dir=exp_dir / "logs",
            enable_console=cfg["LOG"].get("TIMING_CONSOLE", True),
            enable_file=cfg["LOG"].get("TIMING_FILE", True),
            console_update_interval=cfg["LOG"].get("TIMING_CONSOLE_INTERVAL", 10),
            file_flush_interval=cfg["LOG"].get("TIMING_FILE_INTERVAL", 50),
        )
        set_global_timing_logger(timing_logger)

    # Dataset & Loader
    data_root = cfg["DATA"]["ROOT"]
    
    # Check preprocessing configuration and auto-prepare if enabled
    preprocessing_cfg = cfg["DATA"].get("PREPROCESSING", {})
    if preprocessing_cfg.get("AUTO_PREPARE", False) and is_main_process:
        dataset_type = preprocessing_cfg.get("DATASET_TYPE", "gopro")
        force_recompute = preprocessing_cfg.get("FORCE_RECOMPUTE", False)
        
        # Prepare kwargs based on dataset type
        kwargs = {}
        voxel_cfg = preprocessing_cfg.get("VOXEL", {})
        if voxel_cfg:
            kwargs["num_bins"] = voxel_cfg.get("NUM_BINS", 32)
        
        if dataset_type == "gopro":
            gopro_cfg = preprocessing_cfg.get("GOPRO", {})
            kwargs.update({
                "spike_frames": gopro_cfg.get("SPIKE_TEMPORAL_FRAMES", 10),
                "spike_height": gopro_cfg.get("SPIKE_HEIGHT", 396),
                "spike_width": gopro_cfg.get("SPIKE_WIDTH", 640),
            })
        elif dataset_type == "x4k":
            x4k_cfg = preprocessing_cfg.get("X4K", {})
            kwargs.update({
                "fps": x4k_cfg.get("FPS", 1000),
                "exposure_frames": x4k_cfg.get("EXPOSURE_FRAMES", 33),
            })
        
        # Get paths - support both absolute and relative
        data_root_path = Path(data_root)
        if not data_root_path.is_absolute():
            data_root_path = REPO_ROOT / data_root_path
        
        # Create preprocessor
        try:
            print(f"\n[train] Checking preprocessing status for {dataset_type} dataset...")
            preprocessor = get_preprocessor(
                dataset_type=dataset_type,
                data_root=data_root_path.parent / "raw" / dataset_type,  # Assume raw data is in ../raw/
                output_root=data_root_path,
                config_path=Path(args.config),
                **kwargs,
            )
            
            # Check if preprocessing is needed
            is_ready = preprocessor.check_ready()
            
            if not is_ready or force_recompute:
                print(f"[train] Dataset not ready. Running preprocessing pipeline...")
                preprocessor.prepare(force=force_recompute)
                print(f"[train] Preprocessing complete!")
            else:
                print(f"[train] Dataset already preprocessed and ready.")
        except Exception as e:
            print(f"[train] Warning: Preprocessing check/prepare failed: {e}")
            print(f"[train] Continuing with existing data...")
    
    # Wait for main process to complete preprocessing
    if world_size > 1:
        dist.barrier()
    clip_len = int(cfg["DATA"]["CLIP_LEN"])  # baseline: 5
    crop_size = int(cfg["DATA"].get("CROP_SIZE", 256)) if cfg["DATA"].get("CROP_SIZE") else None

    # Prepare image extensions and align log paths
    image_exts = set(cfg["DATA"].get("IMAGE_EXTS", [".png", ".jpg", ".jpeg", ".bmp"]))
    align_log_paths = cfg["DATA"].get("ALIGN_LOG_PATHS", ["outputs/logs/align_x4k1000fps.txt"])
    
    train_set = SpikeDeblurDataset(
        root=data_root,
        split=cfg["DATA"].get("TRAIN_SPLIT", "train"),
        clip_length=clip_len,
        voxel_dirname=cfg["DATA"].get("VOXEL_CACHE_DIRNAME", "spike_vox"),
        crop_size=crop_size,
        spike_dir=cfg["DATA"].get("SPIKE_DIR", "spike"),
        num_voxel_bins=int(cfg["DATA"].get("NUM_VOXEL_BINS", 5)),
        use_precomputed_voxels=bool(cfg["DATA"].get("USE_PRECOMPUTED_VOXELS", True)),
        image_exts=image_exts,
        align_log_paths=align_log_paths,
        use_ram_cache=bool(cfg["DATA"].get("USE_RAM_CACHE", False)),
        cache_size_gb=float(cfg["DATA"].get("CACHE_SIZE_GB", 50.0)),
    )
    
    # Log memory usage after dataset creation
    log_memory_usage(rank, "After dataset creation")
    
    # 计算最优的DataLoader worker配置
    worker_config = get_optimal_num_workers(cfg, world_size)
    
    # 优化DataLoader配置
    batch_size = int(cfg["TRAIN"].get("BATCH_SIZE", 4))
    
    # 获取DataLoader配置（兼容旧配置）
    dataloader_cfg = cfg.get("DATALOADER", {})
    
    # 优先使用新的DATALOADER配置，否则回退到TRAIN配置（向后兼容）
    if "DATALOADER" in cfg:
        # 使用新的配置结构
        train_num_workers = worker_config['train']
        val_num_workers = worker_config['val']
        train_prefetch_factor = dataloader_cfg.get("TRAIN_PREFETCH_FACTOR", 4)
        val_prefetch_factor = dataloader_cfg.get("VAL_PREFETCH_FACTOR", 2)
        pin_memory = dataloader_cfg.get("PIN_MEMORY", True)
        persistent_workers = dataloader_cfg.get("PERSISTENT_WORKERS", True)
    else:
        # 向后兼容：使用TRAIN配置
        train_num_workers = int(cfg["TRAIN"].get("NUM_WORKERS", worker_config['train']))
        val_num_workers = int(cfg["TRAIN"].get("VAL_NUM_WORKERS", worker_config['val']))
        train_prefetch_factor = int(cfg["TRAIN"].get("PREFETCH_FACTOR", 4))
        val_prefetch_factor = int(cfg["TRAIN"].get("VAL_PREFETCH_FACTOR", 2))
        pin_memory = True
        persistent_workers = True if train_num_workers > 0 else False
    
    if is_main_process:
        print(f"[train] ========== 数据加载配置 ==========")
        print(f"[train] CPU核心数: {worker_config['cpu_count']}")
        print(f"[train] 总worker数: {worker_config['total']} (建议值，用于所有{world_size}个GPU)")
        print(f"[train] 训练DataLoader:")
        print(f"[train]   - batch_size: {batch_size} (每GPU)")
        print(f"[train]   - num_workers: {train_num_workers} (每GPU)")
        print(f"[train]   - prefetch_factor: {train_prefetch_factor}")
        print(f"[train]   - pin_memory: {pin_memory}")
        print(f"[train]   - persistent_workers: {persistent_workers}")
        print(f"[train] 验证DataLoader:")
        print(f"[train]   - num_workers: {val_num_workers} (每GPU)")
        print(f"[train]   - prefetch_factor: {val_prefetch_factor}")
        print(f"[train] 总计: {train_num_workers * world_size} 个训练workers 利用CPU资源")
        print(f"[train] ======================================")
    
    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
        sampler=train_sampler,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_spike_deblur_collate,
        drop_last=True,
        prefetch_factor=train_prefetch_factor if train_num_workers > 0 else None,
        persistent_workers=persistent_workers if train_num_workers > 0 else False,
    )

    # Val loader - try multiple split names for robustness
    # Priority: val > test > None (skip validation)
    val_loader = None
    val_split_candidates = [
        cfg["DATA"].get("VAL_SPLIT", "val"),  # User-specified or default "val"
        "val",
        "test",
    ]
    
    # Get validation crop size from config (default to train crop_size to save memory)
    # Set to None in config to use full resolution (requires more GPU memory)
    val_crop_size = cfg["DATA"].get("VAL_CROP_SIZE", crop_size)
    if is_main_process and val_crop_size is not None:
        print(f"[train] Validation crop_size: {val_crop_size} (to save GPU memory during validation)")
    elif is_main_process:
        print(f"[train] WARNING: Validation using full resolution images (may cause OOM)")
    
    for val_split in val_split_candidates:
        val_split_path = Path(data_root) / val_split
        if val_split_path.exists() and any(val_split_path.iterdir()):
            print(f"[train] Using validation split: {val_split}")
            try:
                val_set = SpikeDeblurDataset(
                    root=data_root,
                    split=val_split,
                    clip_length=clip_len,
                    voxel_dirname=cfg["DATA"].get("VOXEL_CACHE_DIRNAME", "spike_vox"),
                    crop_size=val_crop_size,  # Use configured val_crop_size instead of None
                    spike_dir=cfg["DATA"].get("SPIKE_DIR", "spike"),
                    num_voxel_bins=int(cfg["DATA"].get("NUM_VOXEL_BINS", 5)),
                    use_precomputed_voxels=bool(cfg["DATA"].get("USE_PRECOMPUTED_VOXELS", True)),
                    image_exts=image_exts,
                    align_log_paths=align_log_paths,
                    use_ram_cache=False,  # Validation set typically doesn't benefit much from caching
                    cache_size_gb=float(cfg["DATA"].get("CACHE_SIZE_GB", 50.0)),
                )
                if len(val_set) > 0:
                    val_batch_size = int(cfg["TRAIN"].get("VAL_BATCH_SIZE", 1))
                    val_loader = DataLoader(
                        val_set,
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=val_num_workers,
                        pin_memory=pin_memory,
                        collate_fn=safe_spike_deblur_collate,
                        prefetch_factor=val_prefetch_factor if val_num_workers > 0 else None,
                        persistent_workers=persistent_workers if val_num_workers > 0 else False,
                    )
                    break
            except Exception as e:
                print(f"[train] Warning: Failed to load validation split '{val_split}': {e}")
                continue
    
    if val_loader is None:
        print("[train] Warning: No validation split found (tried: val, test). Training without validation.")

    # Model: VRT + Spike wrapper, align with MODEL.VRT_CFG and CHANNELS_PER_SCALE if available
    use_spike = bool(cfg.get("MODEL", {}).get("USE_SPIKE", True))
    # CRITICAL: Default changed from [120]*7 to [96,96,96,96] to match typical config
    channels_per_scale = cfg.get("MODEL", {}).get("CHANNELS_PER_SCALE", [96, 96, 96, 96])
    tsa_heads = int(cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("HEADS", 4))
    fuse_heads = int(cfg.get("MODEL", {}).get("FUSE", {}).get("HEADS", 4))
    
    # Load optional VRT config for window/img_size if file exists
    img_size_cfg = [clip_len, crop_size or 256, crop_size or 256]
    window_size_cfg = [clip_len, 8, 8]
    vrt_cfg_path = cfg.get("MODEL", {}).get("VRT_CFG")
    if vrt_cfg_path is not None:
        try:
            with open(vrt_cfg_path, "r", encoding="utf-8") as vf:
                vrt_opt = yaml.safe_load(vf)
            # Best-effort mapping; keys depend on upstream. Fallbacks remain if not present.
            if isinstance(vrt_opt, dict):
                if "img_size" in vrt_opt:
                    img_size_cfg = list(vrt_opt["img_size"])  # expected [T,H,W]
                if "window_size" in vrt_opt:
                    window_size_cfg = list(vrt_opt["window_size"])  # expected [T,h,w]
        except Exception:
            print(f"[train] Warning: Could not load MODEL.VRT_CFG: {vrt_cfg_path}. Using defaults.")

    # Configure VRT with channels matching CHANNELS_PER_SCALE
    # Per v2 guidelines: stage1-4 must match Spike encoder channels
    # Keep depths at reasonable values for memory efficiency
    # Stage 1-4 (encoder): channels from config, Stage 5-7: keep consistent
    stage_channels = channels_per_scale  # [96, 96, 96, 96] for stage 1-4
    # Extend to 13 values for all RSTB blocks (stage 1-7 use first 7, stage 8 uses rest)
    embed_dims_cfg = stage_channels + [stage_channels[-1]] * 3 + [stage_channels[-1]] * 6
    
    if is_main_process:
        print(f"[train] channels_per_scale from config: {channels_per_scale}")
        print(f"[train] Final embed_dims_cfg for VRT: {embed_dims_cfg} (length={len(embed_dims_cfg)})")
    
    # Read gradient checkpointing settings from config (default: True for memory efficiency)
    vrt_cfg = cfg.get("MODEL", {}).get("VRT", {})
    use_checkpoint_attn = vrt_cfg.get("USE_CHECKPOINT_ATTN", True)
    use_checkpoint_ffn = vrt_cfg.get("USE_CHECKPOINT_FFN", True)
    
    if is_main_process:
        print(f"[train] VRT gradient checkpointing: attn={use_checkpoint_attn}, ffn={use_checkpoint_ffn}")
    
    vrt = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=img_size_cfg,
        window_size=window_size_cfg,
        embed_dims=embed_dims_cfg,  # Match Spike encoder channels for stage 1-4
        use_checkpoint_attn=use_checkpoint_attn,
        use_checkpoint_ffn=use_checkpoint_ffn,
    )
    if use_spike:
        # K is fixed at 32 per v2 guidelines
        spike_bins = int(cfg["DATA"]["K"])
        
        # Spike encoder strides (optional, uses defaults if not specified)
        temporal_strides = cfg.get("MODEL", {}).get("SPIKE_ENCODER", {}).get("TEMPORAL_STRIDES")
        spatial_strides = cfg.get("MODEL", {}).get("SPIKE_ENCODER", {}).get("SPATIAL_STRIDES")
        
        # Spike TSA parameters
        tsa_dropout = float(cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("DROPOUT", 0.0))
        tsa_mlp_ratio = int(cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("MLP_RATIO", 2))
        
        # Fusion parameters
        fuse_dropout = float(cfg.get("MODEL", {}).get("FUSE", {}).get("DROPOUT", 0.0))
        fuse_mlp_ratio = int(cfg.get("MODEL", {}).get("FUSE", {}).get("MLP_RATIO", 2))
        
        # Chunking configurations for attention modules
        tsa_chunk_cfg = cfg.get("MODEL", {}).get("SPIKE_TSA")
        fuse_chunk_cfg = cfg.get("MODEL", {}).get("FUSE")
        
        model = VRTWithSpike(
            vrt_backbone=vrt, 
            spike_bins=spike_bins, 
            channels_per_scale=channels_per_scale,
            temporal_strides=temporal_strides,
            spatial_strides=spatial_strides,
            tsa_heads=tsa_heads,
            tsa_dropout=tsa_dropout,
            tsa_mlp_ratio=tsa_mlp_ratio,
            tsa_chunk_cfg=tsa_chunk_cfg,
            fuse_heads=fuse_heads,
            fuse_dropout=fuse_dropout,
            fuse_mlp_ratio=fuse_mlp_ratio,
            fuse_chunk_cfg=fuse_chunk_cfg,
        )
    else:
        model = vrt
    
    # Move model to GPU
    model = model.to(device)
    
    # Enable Flash Attention for faster attention computation
    from src.utils.flash_attention import apply_flash_attention_to_vrt, check_flash_attention_available
    is_flash_available, flash_method = check_flash_attention_available()
    if is_flash_available:
        if is_main_process:
            print(f"[train] Enabling Flash Attention (method: {flash_method})...")
        try:
            # 获取 VRT backbone (可能是 model.vrt 或 model 本身)
            vrt_to_patch = model.vrt if hasattr(model, 'vrt') else model
            apply_flash_attention_to_vrt(vrt_to_patch)
            if is_main_process:
                print("[train] Flash Attention enabled for VRT")
        except Exception as e:
            if is_main_process:
                print(f"[train] Warning: Failed to enable Flash Attention: {e}")
    else:
        if is_main_process:
            print(f"[train] Flash Attention not available, using standard attention")
    
    # Enable channels_last memory format for Conv2D layers (better performance on Ampere GPUs)
    # Note: We apply this to Conv2D layers only, as the model may contain Conv3D or other non-4D tensors
    channels_last_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                module.to(memory_format=torch.channels_last)
                channels_last_count += 1
            except Exception:
                pass  # Skip layers that don't support channels_last
    if is_main_process:
        if channels_last_count > 0:
            print(f"[train] Enabled channels_last for {channels_last_count} Conv2D layers")
        else:
            print("[train] Warning: No Conv2D layers found for channels_last optimization")
    
    # Enable torch.compile for faster training (PyTorch 2.0+)
    compile_model = cfg.get("TRAIN", {}).get("COMPILE_MODEL", True)
    if compile_model and hasattr(torch, 'compile'):
        if is_main_process:
            print("[train] Compiling model with torch.compile...")
        try:
            model = torch.compile(model, fullgraph=False, dynamic=True)
            if is_main_process:
                print("[train] Model compilation successful")
        except Exception as e:
            if is_main_process:
                print(f"[train] Warning: torch.compile failed: {e}")
    elif is_main_process and not hasattr(torch, 'compile'):
        print("[train] Warning: torch.compile not available (PyTorch < 2.0)")
    
    # Enable multi-GPU training with DistributedDataParallel
    # Note: find_unused_parameters=False is required for gradient checkpointing compatibility
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        # Adjust batch size info for logging
        effective_batch_size = batch_size * world_size
        if is_main_process:
            print(f"[train] Effective batch size: {effective_batch_size} ({batch_size} per GPU)")
            # Estimate GPU memory usage dynamically
            # Model parameters: ~5-8GB, Optimizer (Adam): ~2x model, Gradients: ~1x model
            # Activations (with grad checkpoint): ~5-10GB, Loss models (VGG/LPIPS): ~2-3GB
            # Total rough estimate: (model_size * 4) + 10GB for activations/losses
            model_params = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
            estimated_gpu_mem = model_params * 4 + 10  # 4x for params+optimizer+grads, +10GB for rest
            print(f"[train] Estimated GPU memory per GPU: ~{estimated_gpu_mem:.1f}GB (model: {model_params:.1f}GB, optimizer+grads: {model_params*3:.1f}GB, activations+losses: ~10GB)")
    
    # Enable TF32 for better performance on A6000
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set matmul precision for better performance
        torch.set_float32_matmul_precision('high')
        if is_main_process:
            print("[train] TF32 enabled for faster training on Ampere GPUs")
            print("[train] Float32 matmul precision set to 'high'")
    
    # Get the actual model for parameter access (unwrap DDP if needed)
    model_module = model.module if isinstance(model, DDP) else model

    # Losses
    charbonnier = CharbonnierLoss(delta=float(cfg["LOSS"]["CHARBONNIER"]["DELTA"]))
    vgg = VGGPerceptualLoss(layers=cfg["LOSS"]["VGG_PERCEPTUAL"]["LAYERS"]).to(device)
    w_vgg = float(cfg["LOSS"]["VGG_PERCEPTUAL"]["WEIGHT"])  # default 0.1

    # Optimizer & Scheduler (use model_module for parameters)
    # Support 8-bit Adam optimizer for memory efficiency
    optim_type = cfg["TRAIN"]["OPTIM"].get("TYPE", "adamw").lower()
    lr = float(cfg["TRAIN"]["OPTIM"]["LR"])
    betas = tuple(cfg["TRAIN"]["OPTIM"]["BETAS"])
    weight_decay = float(cfg["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
    
    if optim_type == "adamw8bit":
        try:
            import bitsandbytes as bnb
            optim = bnb.optim.AdamW8bit(
                model_module.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
            if is_main_process:
                print(f"[train] Using 8-bit AdamW optimizer (saves ~75% optimizer memory)")
                # Calculate expected optimizer state memory savings
                param_count = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
                fp32_optim_mem = param_count * 8 / (1024**3)  # 8 bytes per param (momentum + variance in FP32)
                int8_optim_mem = param_count * 2 / (1024**3)  # 2 bytes per param (momentum + variance in INT8)
                saved_mem = fp32_optim_mem - int8_optim_mem
                print(f"[train]   Optimizer state memory: {int8_optim_mem:.2f}GB (vs {fp32_optim_mem:.2f}GB for FP32)")
                print(f"[train]   Memory saved: {saved_mem:.2f}GB ({saved_mem/fp32_optim_mem*100:.1f}%)")
        except ImportError:
            if is_main_process:
                print("[train] Warning: bitsandbytes not installed. Install with: pip install bitsandbytes")
                print("[train] Falling back to standard AdamW")
            optim = torch.optim.AdamW(
                model_module.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
    else:
        optim = torch.optim.AdamW(
            model_module.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        if is_main_process:
            print(f"[train] Using standard AdamW optimizer")
    
    # Support both EPOCHS and TOTAL_STEPS (EPOCHS takes priority per v2 doc)
    grad_accum_steps = int(cfg["TRAIN"].get("GRADIENT_ACCUMULATION_STEPS", 1))
    if "EPOCHS" in cfg["TRAIN"]:
        epochs = int(cfg["TRAIN"]["EPOCHS"])
        steps_per_epoch = len(train_loader) // grad_accum_steps
        total_steps = epochs * steps_per_epoch
        if is_main_process:
            print(f"[train] Using EPOCHS mode: {epochs} epochs × {steps_per_epoch} steps/epoch = {total_steps} total steps")
    else:
        total_steps = int(cfg["TRAIN"]["TOTAL_STEPS"])
        if is_main_process:
            print(f"[train] Using TOTAL_STEPS mode: {total_steps} steps")
    
    warmup = int(cfg["TRAIN"].get("SCHED", {}).get("WARMUP_STEPS", 0))
    # Optional warmup: if warmup>0, use LinearLR -> CosineAnnealingLR, else only Cosine
    if warmup > 0 and warmup < total_steps:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        scheduler = SequentialLR(
            optim,
            schedulers=[
                LinearLR(optim, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(optim, T_max=total_steps - warmup),
            ],
            milestones=[warmup],
        )
    else:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optim, T_max=total_steps)

    # Use mixed precision training for better performance
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp and is_main_process:
        print("[train] Mixed precision training (AMP) enabled")

    # LPIPS for validation
    try:
        import lpips  # type: ignore
        lpips_model = lpips.LPIPS(net='alex').to(device)
    except Exception:
        lpips_model = None

    # Initialize training state
    step = 0
    epoch = 0
    best_psnr = -1.0
    
    # Determine checkpoint path: command line > config file > none
    resume_path = None
    if args.resume is not None:
        # Command line argument takes priority
        resume_path = Path(args.resume)
        if is_main_process:
            print(f"[train] Using checkpoint from command line: {resume_path}")
    elif cfg["TRAIN"].get("RESUME_FROM_CHECKPOINT", False):
        # Check config file settings
        config_ckpt_path = cfg["TRAIN"].get("CHECKPOINT_PATH", None)
        if config_ckpt_path is not None:
            # Use specified checkpoint path from config
            resume_path = Path(config_ckpt_path)
            if is_main_process:
                print(f"[train] Using checkpoint from config: {resume_path}")
        else:
            # Auto-find latest checkpoint
            resume_path = find_latest_checkpoint(exp_dir / "checkpoints")
            if resume_path is not None and is_main_process:
                print(f"[train] Auto-found latest checkpoint: {resume_path}")
            elif is_main_process:
                print(f"[train] RESUME_FROM_CHECKPOINT=true but no checkpoint found. Starting from scratch.")
    
    # Resume from checkpoint if we have a path
    if resume_path is not None:
        resume_path = Path(resume_path)
        if resume_path.exists():
            if is_main_process:
                print(f"[train] Loading checkpoint from {resume_path}")
            
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            
            # Load model state
            model_module.load_state_dict(checkpoint['state_dict'])
            if is_main_process:
                print(f"[train] ✓ Model weights loaded")
            
            # Load training state
            if 'step' in checkpoint:
                step = checkpoint['step']
                if is_main_process:
                    print(f"[train] ✓ Resuming from step {step}")
            
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
            
            if 'best_psnr' in checkpoint:
                best_psnr = checkpoint['best_psnr']
                if is_main_process:
                    print(f"[train] ✓ Best PSNR so far: {best_psnr:.4f}")
            
            # Load optimizer state if available
            if 'optimizer' in checkpoint:
                optim.load_state_dict(checkpoint['optimizer'])
                if is_main_process:
                    print(f"[train] ✓ Optimizer state loaded")
            
            # Load scheduler state if available
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                if is_main_process:
                    print(f"[train] ✓ Scheduler state loaded")
            
            # Load scaler state if available
            if 'scaler' in checkpoint and use_amp:
                scaler.load_state_dict(checkpoint['scaler'])
                if is_main_process:
                    print(f"[train] ✓ AMP scaler state loaded")
            
            if is_main_process:
                print(f"[train] Resume complete. Continuing from step {step}/{total_steps}")
        else:
            if is_main_process:
                print(f"[train] Warning: Checkpoint {resume_path} not found. Starting from scratch.")
    
    model.train()
    val_every = int(cfg["LOG"].get("VAL_EVERY_STEPS", 1000))
    # Track the latest validation metrics to reference in epoch summary
    last_val_metrics = None  # type: ignore
    
    # Training performance monitoring
    # Use deque with fixed maxlen to prevent unbounded memory growth
    from collections import deque
    batch_times = deque(maxlen=100)  # Keep only last 100 samples
    data_times = deque(maxlen=100)   # Keep only last 100 samples
    start_time = time.time()
    last_progress_t = start_time
    
    # Setup profiler if requested
    profiler = None
    if args.profile and is_main_process:
        (exp_dir / "prof").mkdir(parents=True, exist_ok=True)
        prof_dir = exp_dir / "prof"
        if is_main_process:
            print(f"[train] Profiling enabled for {args.profile_steps} steps, output: {prof_dir}")
        
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=args.profile_steps,
                repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    
    if is_main_process:
        print(f"[train] Starting training loop...", flush=True)
        print(f"[train] Total steps: {total_steps}, Validation every: {val_every} steps", flush=True)
        print(f"[train] Gradient accumulation steps: {grad_accum_steps}", flush=True)
    
    save_every_epoch = cfg["LOG"].get("CHECKPOINT_SAVE_EVERY_EPOCH", False)
    keep_last_n = cfg["LOG"].get("CHECKPOINT_KEEP_LAST_N", 0)

    while step < total_steps:
        # Set epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # Initialize epoch-level aggregators
        epoch_start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_charb_sum = 0.0
        epoch_vgg_sum = 0.0
        epoch_steps = 0
        epoch_first_step = step
        
        for micro_step, batch in enumerate(train_loader):
            # Memory ceiling check (every N batches)
            if micro_step % MEMORY_CHECK_INTERVAL == 0:
                needs_cleanup, mem_msg = check_and_manage_memory(step, rank, train_set)
            
            data_load_time = time.time()
            
            # Non-blocking transfer for better GPU utilization
            blur = batch["blur"].to(device, non_blocking=True)           # (B,T,3,H,W)
            sharp = batch["sharp"].to(device, non_blocking=True)         # (B,T,3,H,W)
            spike_vox = batch["spike_vox"].to(device, non_blocking=True) # (B,T,K,H,W)
            
            data_times.append(time.time() - data_load_time)
            batch_start_time = time.time()

            # Forward pass with gradient accumulation
            # For DDP: only sync gradients on the last accumulation step
            is_accum_step = (micro_step + 1) % grad_accum_steps != 0
            sync_context = model.no_sync() if isinstance(model, DDP) and is_accum_step else contextlib.nullcontext()
            
            with sync_context:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    recon = model(blur, spike_vox) if use_spike else model(blur)
                    
                    # Output range monitoring (diagnostic only, no modification)
                    if is_main_process and step % 50 == 0:
                        with torch.no_grad():
                            if (recon < 0).any() or (recon > 1).any():
                                outlier_count = ((recon < 0) | (recon > 1)).sum().item()
                                recon_min = recon.min().item()
                                recon_max = recon.max().item()
                                print(f"[WARNING] Step {step}: {outlier_count} pixels outside [0,1], range=[{recon_min:.3f}, {recon_max:.3f}]", flush=True)
                    
                    l_charb = charbonnier(recon, sharp)
                    l_vgg = vgg(recon, sharp)
                    loss = (l_charb + w_vgg * l_vgg) / grad_accum_steps  # Scale loss for accumulation

                scaler.scale(loss).backward()
                
                # NaN/Inf detection after backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process:
                        print(f"[WARNING] Step {step}: NaN/Inf detected in loss! loss={loss.item()}, l_charb={l_charb.item()}, l_vgg={l_vgg.item()}", flush=True)
                        # Check gradients for diagnostic purposes
                        nan_grad_count = 0
                        for name, param in model_module.named_parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                nan_grad_count += 1
                                if nan_grad_count <= 5:  # Only print first 5 to avoid spam
                                    print(f"  -> NaN/Inf in gradient: {name}", flush=True)
                        if nan_grad_count > 5:
                            print(f"  -> ... and {nan_grad_count - 5} more parameters with NaN/Inf gradients", flush=True)
                    # Skip this batch to prevent contamination
                    optim.zero_grad(set_to_none=True)
                    continue
            
            # Only update weights every grad_accum_steps
            if (micro_step + 1) % grad_accum_steps == 0:
                # Unscale before gradient clipping for better numerical stability
                scaler.unscale_(optim)
                max_grad_norm = float(cfg["TRAIN"].get("MAX_GRAD_NORM", 1.0))
                torch.nn.utils.clip_grad_norm_(model_module.parameters(), max_norm=max_grad_norm)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                
                batch_times.append(time.time() - batch_start_time)
                step += 1
                # Aggregate epoch-level metrics using unscaled loss (filter NaN/Inf)
                actual_loss = loss.item() * grad_accum_steps
                if not (math.isnan(actual_loss) or math.isinf(actual_loss)):
                    epoch_loss_sum += float(actual_loss)
                    epoch_charb_sum += float(l_charb.item())
                    epoch_vgg_sum += float(l_vgg.item())
                    epoch_steps += 1
                
                # Step profiler after each training step
                if profiler is not None:
                    profiler.step()
                    # Stop profiling after specified steps
                    if step >= args.profile_steps + 10:
                        profiler.stop()
                        profiler = None
                        if is_main_process:
                            print(f"[train] Profiling completed at step {step}")
                
                # Step timing logger
                if timing_logger is not None:
                    timing_logger.step()
            else:
                # Track batch time even for accumulation steps
                batch_times.append(time.time() - batch_start_time)
            if is_main_process and writer and step % 50 == 0 and (micro_step + 1) % grad_accum_steps == 0:
                # Un-scale loss for logging (multiply back by grad_accum_steps)
                actual_loss = loss.item() * grad_accum_steps
                # Only log valid (non-NaN/Inf) values to TensorBoard
                if not (math.isnan(actual_loss) or math.isinf(actual_loss)):
                    writer.add_scalar('train/loss', actual_loss, step)
                    writer.add_scalar('train/charb', float(l_charb.item()), step)
                    writer.add_scalar('train/vgg', float(l_vgg.item()), step)
                writer.add_scalar('train/lr', float(scheduler.get_last_lr()[0]), step)
                
                # Add GPU memory usage monitoring
                if torch.cuda.is_available():
                    for i in range(num_gpus):
                        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        writer.add_scalar(f'gpu/memory_allocated_gb_gpu{i}', mem_allocated, step)
                        writer.add_scalar(f'gpu/memory_reserved_gb_gpu{i}', mem_reserved, step)
                wandb_logger.log_metrics(
                    {
                        'train/loss': actual_loss,
                        'train/charb': float(l_charb.item()),
                        'train/vgg': float(l_vgg.item()),
                        'train/lr': float(scheduler.get_last_lr()[0]),
                    },
                    step=step,
                )
                
                # Enhanced logging with GPU info and throughput
                gpu_mem_str = ""
                throughput_str = ""
                if torch.cuda.is_available() and num_gpus > 0:
                    mem_gb = torch.cuda.memory_allocated(local_rank) / 1024**3
                    gpu_mem_str = f" | GPU{local_rank}_mem={mem_gb:.1f}GB"
                
                # Calculate throughput
                if len(batch_times) >= 10:
                    # Convert deque to list for slicing (deque doesn't support slice notation)
                    avg_batch_time = np.mean(list(batch_times)[-50:])
                    avg_data_time = np.mean(list(data_times)[-50:])
                    samples_per_sec = (batch_size * world_size if world_size > 1 else batch_size) / avg_batch_time
                    throughput_str = f" | {samples_per_sec:.1f} samples/s | data_time={avg_data_time*1000:.1f}ms"
                    writer.add_scalar('perf/samples_per_sec', samples_per_sec, step)
                    writer.add_scalar('perf/batch_time_ms', avg_batch_time * 1000, step)
                    writer.add_scalar('perf/data_time_ms', avg_data_time * 1000, step)
                    wandb_logger.log_metrics(
                        {
                            'perf/samples_per_sec': float(samples_per_sec),
                            'perf/batch_time_ms': float(avg_batch_time * 1000),
                            'perf/data_time_ms': float(avg_data_time * 1000),
                        },
                        step=step,
                    )
                
                print(f"step {step}/{total_steps} | loss={actual_loss:.4f} charb={l_charb.item():.4f} vgg={l_vgg.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}{gpu_mem_str}{throughput_str}", flush=True)
            
            # Periodic memory monitoring (every 100 steps)
            if is_main_process and step % 100 == 0 and (micro_step + 1) % grad_accum_steps == 0:
                log_memory_usage(rank, f"Step {step}")
                # Log GPU memory to TensorBoard for tracking memory leaks
                if torch.cuda.is_available() and writer:
                    for i in range(num_gpus):
                        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                        mem_max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                        writer.add_scalar(f'memory/gpu{i}_allocated_gb', mem_allocated, step)
                        writer.add_scalar(f'memory/gpu{i}_reserved_gb', mem_reserved, step)
                        writer.add_scalar(f'memory/gpu{i}_max_allocated_gb', mem_max_allocated, step)
            
            # Periodic cache clearing to prevent unbounded growth
            if step > 0 and step % CACHE_CLEAR_INTERVAL == 0 and (micro_step + 1) % grad_accum_steps == 0:
                if is_main_process:
                    print(f"\n[Memory Management] Clearing cache at step {step}", flush=True)
                # Clear dataset cache
                if train_set is not None and hasattr(train_set, 'clear_cache'):
                    train_set.clear_cache()
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Garbage collection
                gc.collect()
                if is_main_process:
                    mem = psutil.virtual_memory() if HAS_PSUTIL else None
                    if mem:
                        print(f"    ✓ Cache cleared, memory: {mem.used/(1024**3):.1f}GB/{mem.total/(1024**3):.1f}GB", flush=True)

            # periodic validation (skip step 0 to avoid duplicate saves)
            # ALL processes participate in validation to avoid DDP sync issues
            if step > 0 and step % val_every == 0:
                # Save checkpoint only on main process
                if is_main_process:
                    # Use model_module to get unwrapped state_dict
                    ckpt_last = exp_dir / "checkpoints" / "last.pth"
                    torch.save({'step': step, 'state_dict': model_module.state_dict()}, ckpt_last)
                    wandb_logger.log_checkpoint(ckpt_last)
                
                # ALL processes participate in validation
                if val_loader is not None:
                    # CRITICAL: Clear GPU cache before validation to prevent OOM
                    # Training may have accumulated fragmented memory over 1000 steps
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    if is_main_process:
                        print(f"[val] Cleared GPU cache before validation at step {step}", flush=True)
                    
                    model.eval()
                    psnr_rgb_list = []
                    psnr_y_list = []
                    ssim_list = []
                    lpips_list = []
                    with torch.no_grad():
                        for val_idx, vbatch in enumerate(val_loader):
                            v_blur = vbatch["blur"].to(device, non_blocking=True)
                            v_sharp = vbatch["sharp"].to(device, non_blocking=True)
                            v_spike = vbatch["spike_vox"].to(device, non_blocking=True)
                            v_recon = model(v_blur, v_spike) if use_spike else model(v_blur)
                            v_recon = torch.clamp(v_recon, 0.0, 1.0)
                            
                            # Compute all metrics in one pass to avoid re-iterating
                            # Use .detach() and .item() to ensure no gradient graph is retained
                            psnr_rgb_list.append(compute_psnr(v_recon.detach(), v_sharp.detach()))
                            
                            y_recon = rgb_to_y(v_recon.detach())
                            y_sharp = rgb_to_y(v_sharp.detach())
                            psnr_y_list.append(compute_psnr(y_recon, y_sharp))
                            
                            # SSIM on Y channel
                            try:
                                ssim_val = compute_ssim(y_recon, y_sharp)
                                ssim_list.append(ssim_val)
                            except Exception:
                                pass  # Skip SSIM for this batch if it fails
                            
                            if lpips_model is not None:
                                # merge (B,T) into batch for lpips expecting (N,3,H,W)
                                nbt = v_recon.shape[0] * v_recon.shape[1]
                                x_lp = v_recon.detach().reshape(nbt, 3, *v_recon.shape[-2:])
                                y_lp = v_sharp.detach().reshape(nbt, 3, *v_sharp.shape[-2:])
                                lp = lpips_model(x_lp, y_lp).mean().item()
                                lpips_list.append(lp)
                                # Clear LPIPS tensors immediately
                                del x_lp, y_lp
                            
                            # Clear intermediate tensors to free GPU memory IMMEDIATELY
                            del v_recon, v_blur, v_sharp, v_spike, y_recon, y_sharp
                            
                            # Periodic GPU cache clearing during validation to prevent accumulation
                            # Clear every 5 batches to balance performance and memory
                            if (val_idx + 1) % 5 == 0:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                    
                    # Aggregate metrics across all GPUs using all_reduce
                    psnr_rgb = float(np.mean(psnr_rgb_list)) if psnr_rgb_list else 0.0
                    psnr_y = float(np.mean(psnr_y_list)) if psnr_y_list else 0.0
                    ssim_y = float(np.mean(ssim_list)) if ssim_list else 0.0
                    lpips_val = float(np.mean(lpips_list)) if lpips_list else 0.0
                    
                    # In multi-GPU setup, average metrics across all ranks
                    if world_size > 1:
                        metrics_tensor = torch.tensor(
                            [psnr_rgb, psnr_y, ssim_y, lpips_val, float(len(psnr_rgb_list))],
                            device=device
                        )
                        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                        # Average by total number of samples across all GPUs
                        total_samples = metrics_tensor[4].item()
                        if total_samples > 0:
                            psnr_rgb = (metrics_tensor[0] / world_size).item()
                            psnr_y = (metrics_tensor[1] / world_size).item()
                            ssim_y = (metrics_tensor[2] / world_size).item()
                            lpips_val = (metrics_tensor[3] / world_size).item()
                    
                    # Only main process logs metrics and saves checkpoints
                    if is_main_process:
                        if writer:
                            writer.add_scalar('val/psnr_rgb', psnr_rgb, step)
                            writer.add_scalar('val/psnr_y', psnr_y, step)
                            writer.add_scalar('val/ssim_y', ssim_y, step)
                            writer.add_scalar('val/lpips', lpips_val, step)
                        wandb_logger.log_metrics(
                            {
                                'val/psnr_rgb': psnr_rgb,
                                'val/psnr_y': psnr_y,
                                'val/ssim_y': ssim_y,
                                'val/lpips': lpips_val,
                            },
                            step=step,
                        )
                        wandb_logger.set_summary({'val/best_psnr_rgb': best_psnr})

                    # save best checkpoint only on main process
                    if is_main_process and psnr_rgb > best_psnr:
                        best_psnr = psnr_rgb
                        # First, construct name
                        best_ckpt_name = f"epoch={epoch:03d}-val_psnr={best_psnr:.2f}.pth"
                        ckpt_best = exp_dir / "checkpoints" / best_ckpt_name
                        # Save to ckpt_best
                        best_checkpoint_dict = {
                            'step': step,
                            'epoch': epoch,
                            'best_psnr': best_psnr,
                            'state_dict': model_module.state_dict(),
                            'optimizer': optim.state_dict(),
                            'scheduler': scheduler.state_dict(),
                        }
                        if use_amp:
                            best_checkpoint_dict['scaler'] = scaler.state_dict()
                        torch.save(best_checkpoint_dict, ckpt_best)
                        wandb_logger.log_checkpoint(ckpt_best, name=f"best-epoch{epoch:03d}")
                        # Also create a symlink or copy to "best.pth" for convenience
                        best_link = exp_dir / "checkpoints" / "best.pth"
                        if best_link.exists():
                            best_link.unlink()
                        best_link.symlink_to(best_ckpt_name)  # Use symlink for efficiency
                    
                    # dump metrics snapshot and print - only on main process
                    if is_main_process:
                        metrics_path = exp_dir / "metrics" / f"val_step_{step}.json"
                        with open(metrics_path, 'w', encoding='utf-8') as f:
                            json.dump({'step': step, 'psnr_rgb': psnr_rgb, 'psnr_y': psnr_y, 'ssim_y': ssim_y, 'lpips': lpips_val}, f, ensure_ascii=False, indent=2)
                        print(f"[val] step {step} | PSNR_RGB={psnr_rgb:.2f} PSNR_Y={psnr_y:.2f} SSIM_Y={ssim_y:.4f} LPIPS={lpips_val:.4f}")
                    
                    # store last validation for epoch summary (all processes need this)
                    last_val_metrics = {
                        'step': step,
                        'psnr_rgb': psnr_rgb,
                        'psnr_y': psnr_y,
                        'ssim_y': ssim_y,
                        'lpips': lpips_val,
                    }
                    
                    # Explicitly clear GPU cache after validation to prevent memory accumulation
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    model.train()
                    
                    # Synchronize all processes after validation to avoid deadlock
                    if world_size > 1:
                        dist.barrier()
                elif is_main_process:
                    # No validation available message - only on main process
                    print(f"[train] step {step} | Checkpoint saved (no validation available)", flush=True)
                
                # Ensure all processes are synchronized after validation/checkpoint save
                if world_size > 1:
                    dist.barrier()

            # SOTA-style progress with ETA (non-intrusive, additional to existing prints)
            if is_main_process and (micro_step + 1) % grad_accum_steps == 0 and progress_every > 0 and step % progress_every == 0:
                elapsed = time.time() - start_time
                done_ratio = min(1.0, step / max(1, total_steps))
                eta = (elapsed / max(1e-9, done_ratio) - elapsed) if done_ratio > 0 else 0.0
                # epoch-local progress (approx by step delta)
                epoch_done_steps = step  # cumulative; combined with epoch_first_step at epoch-end
                bar_len = 20
                filled = int(bar_len * done_ratio)
                bar = "#" * filled + "." * (bar_len - filled)
                print(f"[progress] {step}/{total_steps} [{bar}] {done_ratio*100:5.1f}% | elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}", flush=True)

            # periodic visuals
            if step % 10000 == 0:
                v = torch.clamp(recon[0], 0.0, 1.0)  # take first sample (T,3,H,W)
                s = torch.clamp(sharp[0], 0.0, 1.0)
                b = torch.clamp(blur[0], 0.0, 1.0)
                # stack last frame for quick glance: (3,H,W)
                grid = torch.cat([b[-1], v[-1], s[-1]], dim=-1)  # concat width-wise
                import torchvision.utils as vutils
                out_path = exp_dir / "visuals" / f"step_{step:06d}.png"
                vutils.save_image(grid, str(out_path))
            if step >= total_steps:
                break
        # End of epoch: emit epoch-level summary without changing existing step logs
        if is_main_process and epoch_steps > 0:
            epoch_duration = time.time() - epoch_start_time
            avg_loss = epoch_loss_sum / epoch_steps
            avg_charb = epoch_charb_sum / epoch_steps
            avg_vgg = epoch_vgg_sum / epoch_steps
            eff_bs = (batch_size * world_size) if world_size > 1 else batch_size
            samples_per_sec_epoch = (epoch_steps * eff_bs) / epoch_duration if epoch_duration > 0 else 0.0
            lr_now = float(scheduler.get_last_lr()[0])
            # Console summary
            print(f"[epoch] {epoch+1} | steps={epoch_steps} | loss={avg_loss:.4f} (charb={avg_charb:.4f}, vgg={avg_vgg:.4f}) | lr={lr_now:.2e} | time={epoch_duration:.1f}s | throughput={samples_per_sec_epoch:.1f} samples/s", flush=True)
            # Include last validation seen within this epoch range if available
            if last_val_metrics is not None and last_val_metrics.get('step', -1) >= epoch_first_step and last_val_metrics.get('step', -1) <= step:
                print(f"[epoch] {epoch+1} | last_val@step {last_val_metrics['step']} -> PSNR_RGB={last_val_metrics['psnr_rgb']:.2f} PSNR_Y={last_val_metrics['psnr_y']:.2f} SSIM_Y={last_val_metrics['ssim_y']:.4f} LPIPS={last_val_metrics['lpips']:.4f}", flush=True)
            # TensorBoard epoch-level scalars
            if writer:
                writer.add_scalar('epoch/avg_loss', avg_loss, epoch + 1)
                writer.add_scalar('epoch/avg_charb', avg_charb, epoch + 1)
                writer.add_scalar('epoch/avg_vgg', avg_vgg, epoch + 1)
                writer.add_scalar('epoch/throughput_samples_per_sec', samples_per_sec_epoch, epoch + 1)
                writer.add_scalar('epoch/lr', lr_now, epoch + 1)
            # Persist JSON snapshot
            epoch_metrics_path = exp_dir / "metrics" / f"epoch_{epoch+1:03d}.json"
            with open(epoch_metrics_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'epoch': epoch + 1,
                    'start_step': int(epoch_first_step),
                    'end_step': int(step),
                    'avg_loss': float(avg_loss),
                    'avg_charb': float(avg_charb),
                    'avg_vgg': float(avg_vgg),
                    'lr': float(lr_now),
                    'duration_sec': float(epoch_duration),
                    'throughput_samples_per_sec': float(samples_per_sec_epoch),
                    'last_val_in_epoch': last_val_metrics if (last_val_metrics is not None and last_val_metrics.get('step', -1) >= epoch_first_step and last_val_metrics.get('step', -1) <= step) else None,
                }, f, ensure_ascii=False, indent=2)
        
        # Explicit garbage collection to prevent memory accumulation
        # This helps free up memory from completed batches and intermediate tensors
        if is_main_process and (epoch + 1) % 1 == 0:  # Every epoch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Reset peak memory stats to track memory usage per epoch
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
                if is_main_process:
                    print(f"[Memory] Cleared GPU cache and reset peak memory stats at end of epoch {epoch+1}", flush=True)
        
        epoch += 1
        if save_every_epoch and is_main_process:
            checkpoint_dict = {
                'step': step,
                'epoch': epoch,
                'state_dict': model_module.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if use_amp:
                checkpoint_dict['scaler'] = scaler.state_dict()
            ckpt_epoch_path = exp_dir / "checkpoints" / f"epoch_{epoch:03d}.pth"
            torch.save(checkpoint_dict, ckpt_epoch_path)
        
        epoch += 1

    # 运行信息归档
    run_txt = exp_dir / "logs" / f"run_{int(time.time())}.txt"
    try:
        import subprocess
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        git_rev = "unknown"
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg_text = f.read()
    try:
        import pkgutil
        import pkg_resources
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    except Exception:
        pip_freeze = ""
    with open(run_txt, 'w', encoding='utf-8') as f:
        f.write(f"git_rev: {git_rev}\n\n")
        f.write("# config\n")
        f.write(cfg_text + "\n\n")
        f.write("# pip freeze\n")
        f.write(pip_freeze)

    # Stop profiler if still running
    if profiler is not None:
        profiler.stop()
        if is_main_process:
            print("[train] Profiler stopped")
    
    # Close timing logger and print summary
    if timing_logger is not None:
        timing_logger.print_summary()
        timing_logger.close()
        set_global_timing_logger(None)
    
    if writer:
        writer.flush()
        writer.close()
    
    wandb_logger.close()
    
    # Cleanup distributed training
    cleanup_distributed()
    
    if is_main_process:
        print("[train] Done.")


if __name__ == "__main__":
    main()



