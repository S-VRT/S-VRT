import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import yaml

# Add third_party to path for VRT import
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
VRT_ROOT = REPO_ROOT / "third_party" / "VRT"
if str(VRT_ROOT) not in sys.path:
    sys.path.insert(0, str(VRT_ROOT))

from models.network_vrt import VRT  # type: ignore

from src.data.datasets.spike_deblur_dataset import SpikeDeblurDataset
from src.data.collate_fns import safe_spike_deblur_collate
from src.models.integrate_vrt import VRTWithSpike
from src.utils.path_manager import get_config_from_exp_dir, get_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test VRT+Spike baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML config")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # 目录推理接口（与实施指导一致）
    parser.add_argument("--input_blur", type=str, default=None, help="Directory of input blur frames")
    parser.add_argument("--input_spike_vox", type=str, default=None, help="Directory of input spike voxels (.npy)")
    parser.add_argument("--output", type=str, default=None, help="Directory to write outputs")
    # Tile-based inference for large images
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for inference on large images")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Overlap between tiles")
    parser.add_argument("--enable_tiling", action="store_true", help="Force enable tiling even for small images")
    parser.add_argument("--disable_tiling", action="store_true", help="Disable tiling and process full images")
    parser.add_argument("--exp_dir", type=str, default=None, help="Path to experiment directory (overrides config for loading)")
    parser.add_argument("--checkpoint", type=str, default="best", help="Checkpoint to load: filename, 'best', or 'last'")
    return parser.parse_args()


def tile_inference(
    model: torch.nn.Module,
    blur: torch.Tensor,
    spike_vox: torch.Tensor,
    tile_size: int = 256,
    tile_overlap: int = 32,
    device: torch.device = torch.device("cpu"),
    force_tiling: bool = False,
    disable_tiling: bool = False,
) -> torch.Tensor:
    """
    Perform tile-based inference for large images to avoid OOM.
    
    Args:
        model: The model to use for inference
        blur: Blur frames (B, T, C, H, W)
        spike_vox: Spike voxels (B, T, K, H_s, W_s)
        tile_size: Size of each tile
        tile_overlap: Overlap between adjacent tiles
        device: Device to run inference on
        force_tiling: Force tiling even for small images
        disable_tiling: Disable tiling and process full images
    
    Returns:
        Reconstructed frames (B, T, C, H, W)
    """
    # Input validation
    if blur.dim() != 5 or spike_vox.dim() != 5:
        raise ValueError(f"Expected 5D tensors, got blur: {blur.dim()}D, spike_vox: {spike_vox.dim()}D")
    
    B, T, C, H, W = blur.shape
    _, _, K, H_s, W_s = spike_vox.shape
    
    # Validate batch and time dimensions match
    if blur.shape[:2] != spike_vox.shape[:2]:
        raise ValueError(f"Batch and time dimensions must match: blur {blur.shape[:2]} vs spike_vox {spike_vox.shape[:2]}")
    
    # Validate tile parameters
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError(f"tile_overlap must be >= 0 and < tile_size, got {tile_overlap}")
    
    # Debug: print model VRT configuration
    if hasattr(model, 'vrt') and hasattr(model.vrt, 'img_size'):
        print(f"Model VRT img_size: {model.vrt.img_size}")
        print(f"Model VRT window_size: {model.vrt.window_size}")
    
    print(f"tile_inference: blur={blur.shape}, spike_vox={spike_vox.shape}")
    
    # Determine if tiling should be used
    should_tile = not disable_tiling and (force_tiling or H > tile_size or W > tile_size)
    
    if disable_tiling:
        print("Tiling disabled by user, processing full image")
        return model(blur, spike_vox)
    elif not should_tile:
        print("Image small enough, processing directly without tiling")
        return model(blur, spike_vox)
    else:
        print(f"Using tiling: force_tiling={force_tiling}, image_size=({H},{W}), tile_size={tile_size}")
    
    # Calculate tile positions
    stride = tile_size - tile_overlap
    h_tiles = max(1, (H - tile_overlap + stride - 1) // stride)
    w_tiles = max(1, (W - tile_overlap + stride - 1) // stride)
    
    print(f"Tiling: {h_tiles}x{w_tiles} tiles, stride={stride}, tile_size={tile_size}")
    
    # Output accumulator and weight map for blending
    output = torch.zeros(B, T, C, H, W, device=device, dtype=blur.dtype)
    weight = torch.zeros(B, T, 1, H, W, device=device, dtype=blur.dtype)
    
    # Create a weight map for smooth blending (cosine window)
    tile_weight = torch.ones(1, 1, 1, tile_size, tile_size, device=device, dtype=blur.dtype)
    if tile_overlap > 0:
        # Apply cosine taper at the edges for smooth blending
        fade = torch.linspace(0, 1, tile_overlap, device=device, dtype=blur.dtype)
        fade_window = torch.cos((1 - fade) * torch.pi / 2)
        # Top edge
        tile_weight[:, :, :, :tile_overlap, :] *= fade_window.view(1, 1, 1, -1, 1)
        # Bottom edge
        tile_weight[:, :, :, -tile_overlap:, :] *= fade_window.flip(0).view(1, 1, 1, -1, 1)
        # Left edge
        tile_weight[:, :, :, :, :tile_overlap] *= fade_window.view(1, 1, 1, 1, -1)
        # Right edge
        tile_weight[:, :, :, :, -tile_overlap:] *= fade_window.flip(0).view(1, 1, 1, 1, -1)
    
    # Process tiles with progress tracking
    total_tiles = h_tiles * w_tiles
    processed_tiles = 0
    
    for i in range(h_tiles):
        for j in range(w_tiles):
            processed_tiles += 1
            
            # Calculate tile coordinates with boundary checks
            h_start = i * stride
            w_start = j * stride
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)
            
            # Actual tile size (may be smaller at image boundaries)
            h_tile = h_end - h_start
            w_tile = w_end - w_start
            
            # Skip empty tiles
            if h_tile <= 0 or w_tile <= 0:
                continue
            
            # Extract tiles from blur and spike_vox
            blur_tile = blur[:, :, :, h_start:h_end, w_start:w_end]
            
            # Calculate corresponding spike voxel coordinates with proper scaling
            scale_h = H_s / H
            scale_w = W_s / W
            h_s_start = max(0, int(h_start * scale_h))
            w_s_start = max(0, int(w_start * scale_w))
            h_s_end = min(H_s, int(h_end * scale_h))
            w_s_end = min(W_s, int(w_end * scale_w))
            
            # Ensure spike voxel coordinates are valid
            if h_s_end <= h_s_start or w_s_end <= w_s_start:
                print(f"Warning: Invalid spike voxel coordinates for tile ({i},{j})")
                continue
                
            spike_vox_tile = spike_vox[:, :, :, h_s_start:h_s_end, w_s_start:w_s_end]
            
            # Get actual spike voxel tile size
            h_s_tile = h_s_end - h_s_start
            w_s_tile = w_s_end - w_s_start
            
            # Calculate target spike voxel tile size (should be proportional to blur tile_size)
            target_h_s = int(tile_size * scale_h)
            target_w_s = int(tile_size * scale_w)
            
            # Pad to tile_size if necessary
            if h_tile < tile_size or w_tile < tile_size:
                # Use constant padding for 5D tensors
                pad_h = tile_size - h_tile
                pad_w = tile_size - w_tile
                blur_tile = F.pad(blur_tile, (0, pad_w, 0, pad_h), mode='constant', value=0)
            if h_s_tile < target_h_s or w_s_tile < target_w_s:
                # Use constant padding for 5D tensors
                pad_h_s = target_h_s - h_s_tile
                pad_w_s = target_w_s - w_s_tile
                spike_vox_tile = F.pad(spike_vox_tile, (0, pad_w_s, 0, pad_h_s), mode='constant', value=0)
            
            # Debug: print shapes for first tile
            if i == 0 and j == 0:
                print(f"  Tile {i},{j}: blur_tile={blur_tile.shape}, spike_vox_tile={spike_vox_tile.shape}")
            
            # Process tile with error handling
            try:
                with torch.no_grad():
                    recon_tile = model(blur_tile, spike_vox_tile)
            except Exception as e:
                print(f"Error processing tile ({i},{j}): {e}")
                continue
            
            # Remove padding if added
            if h_tile < tile_size or w_tile < tile_size:
                recon_tile = recon_tile[:, :, :, :h_tile, :w_tile]
            
            # Get current tile weight
            current_weight = tile_weight[:, :, :, :h_tile, :w_tile]
            
            # Accumulate output and weights
            output[:, :, :, h_start:h_end, w_start:w_end] += recon_tile * current_weight
            weight[:, :, :, h_start:h_end, w_start:w_end] += current_weight
            
            # Print progress every 10% or for small numbers
            if total_tiles <= 10 or processed_tiles % max(1, total_tiles // 10) == 0:
                print(f"Processed {processed_tiles}/{total_tiles} tiles ({100*processed_tiles/total_tiles:.1f}%)")
    
    # Normalize by weights with numerical stability
    print(f"Completed tile processing: {processed_tiles}/{total_tiles} tiles")
    
    # Check for zero weights (shouldn't happen with proper tiling)
    zero_weight_mask = weight < 1e-8
    if zero_weight_mask.any():
        print(f"Warning: Found {zero_weight_mask.sum().item()} pixels with zero weights")
        # Fill zero weights with small value to avoid division by zero
        weight = torch.where(zero_weight_mask, torch.tensor(1e-8, device=device, dtype=weight.dtype), weight)
    
    # Normalize output
    output = output / weight
    
    # Final validation
    if torch.isnan(output).any() or torch.isinf(output).any():
        print("Warning: Output contains NaN or Inf values")
        output = torch.where(torch.isnan(output) | torch.isinf(output), torch.zeros_like(output), output)
    
    print(f"Tile inference completed successfully. Output shape: {output.shape}")
    return output


def main() -> None:
    args = parse_args()
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        config_path = get_config_from_exp_dir(exp_dir)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: Dict = yaml.safe_load(f)
        checkpoint_path = get_checkpoint_path(exp_dir, args.checkpoint)
    else:
        # old behavior
        with open(args.config, "r", encoding="utf-8") as f:
            cfg: Dict = yaml.safe_load(f)
    
    # Set environment variable for collate_fn to use
    save_dir = cfg["LOG"].get("SAVE_DIR", "outputs")
    os.environ["DEBLUR_SAVE_DIR"] = str(REPO_ROOT / save_dir)

    device = torch.device(args.device)
    data_root = cfg["DATA"]["ROOT"]
    clip_len = int(cfg["DATA"]["CLIP_LEN"])  # baseline: 5
    crop_size = None  # validation full-size

    # Prepare image extensions and align log paths
    image_exts = set(cfg["DATA"].get("IMAGE_EXTS", [".png", ".jpg", ".jpeg", ".bmp"]))
    align_log_paths = cfg["DATA"].get("ALIGN_LOG_PATHS", ["outputs/logs/align_x4k1000fps.txt"])

    use_dir_api = args.input_blur is not None and args.input_spike_vox is not None and args.output is not None
    if not use_dir_api:
        # Dataset/val 分割
        val_set = SpikeDeblurDataset(
            root=data_root,
            split=cfg["DATA"].get("VAL_SPLIT", "val"),
            clip_length=clip_len,
            voxel_dirname=cfg["DATA"].get("VOXEL_CACHE_DIRNAME", "spike_vox"),
            crop_size=crop_size,
            spike_dir=cfg["DATA"].get("SPIKE_DIR", "spike"),
            num_voxel_bins=int(cfg["DATA"].get("NUM_VOXEL_BINS", 5)),
            use_precomputed_voxels=bool(cfg["DATA"].get("USE_PRECOMPUTED_VOXELS", True)),
            image_exts=image_exts,
            align_log_paths=align_log_paths,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=int(cfg.get("TEST", {}).get("BATCH_SIZE", 1)),
            shuffle=False,
            num_workers=int(cfg.get("TEST", {}).get("NUM_WORKERS", 2)),
            collate_fn=safe_spike_deblur_collate,
        )
    else:
        # 简单的目录 API：按名称对齐 (000000.png ↔ 000000.npy)
        blur_dir = Path(args.input_blur)
        vox_dir = Path(args.input_spike_vox)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Window size for VRT (default 8x8 for deblur task)
    window_size = cfg.get("MODEL", {}).get("WINDOW_SIZE", [clip_len, 8, 8])
    if isinstance(window_size, int):
        window_size = [clip_len, window_size, window_size]
    
    use_spike = bool(cfg.get("MODEL", {}).get("USE_SPIKE", True))
    
    # Image size for VRT (default 256x256 for deblur task)
    img_size_h = int(cfg.get("MODEL", {}).get("IMG_SIZE_H", 256))
    img_size_w = int(cfg.get("MODEL", {}).get("IMG_SIZE_W", 256))
    
    # CRITICAL: Must match training! VRT default is 16
    deform_groups = int(cfg.get("MODEL", {}).get("DEFORM_GROUPS", 16))
    
    # Spike parameters (must be read BEFORE creating VRT to set embed_dims correctly)
    spike_bins = int(cfg["DATA"]["K"])
    
    # CRITICAL: Read CHANNELS_PER_SCALE from config (must match training!)
    # Default changed from [120]*7 to [96]*4 to match typical training config
    channels_per_scale = cfg.get("MODEL", {}).get("CHANNELS_PER_SCALE", [96, 96, 96, 96])
    
    # Debug: Print the configuration being used
    print(f"[test.py] channels_per_scale from config: {channels_per_scale}")
    
    # Configure VRT embed_dims to match training configuration
    # Stage 1-4 use channels_per_scale, stages 5-7 and 8 extend with the last value
    stage_channels = channels_per_scale
    embed_dims_cfg = stage_channels + [stage_channels[-1]] * 3 + [stage_channels[-1]] * 6
    print(f"[test.py] Final embed_dims_cfg for VRT: {embed_dims_cfg} (length={len(embed_dims_cfg)})")

    vrt = VRT(
        upscale=1,
        in_chans=3,
        out_chans=3,
        img_size=[clip_len, img_size_h, img_size_w],
        window_size=window_size,
        embed_dims=embed_dims_cfg,  # CRITICAL: Must match training config
        deformable_groups=deform_groups,
    )
    tsa_heads = int(cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("HEADS", 4))
    fuse_heads = int(cfg.get("MODEL", {}).get("FUSE", {}).get("HEADS", 4))
    
    # Spike encoder strides (optional, uses defaults if not specified)
    temporal_strides = cfg.get("MODEL", {}).get("SPIKE_ENCODER", {}).get("TEMPORAL_STRIDES")
    spatial_strides = cfg.get("MODEL", {}).get("SPIKE_ENCODER", {}).get("SPATIAL_STRIDES")
    
    # Spike TSA parameters
    tsa_dropout = float(cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("DROPOUT", 0.0))
    tsa_mlp_ratio = int(cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("MLP_RATIO", 2))
    
    # Fusion parameters
    fuse_dropout = float(cfg.get("MODEL", {}).get("FUSE", {}).get("DROPOUT", 0.0))
    fuse_mlp_ratio = int(cfg.get("MODEL", {}).get("FUSE", {}).get("MLP_RATIO", 2))
    
    # Build chunk config dictionaries for TSA and Fuse
    tsa_chunk_cfg = {
        "adaptive": cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("ADAPTIVE_CHUNK", True),
        "max_batch_tokens": cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("MAX_BATCH_TOKENS", 49152),
        "chunk_size": cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("CHUNK_SIZE", 64),
        "chunk_shape": cfg.get("MODEL", {}).get("SPIKE_TSA", {}).get("CHUNK_SHAPE", "square"),
    }
    
    fuse_chunk_cfg = {
        "adaptive": cfg.get("MODEL", {}).get("FUSE", {}).get("ADAPTIVE_CHUNK", True),
        "max_batch_tokens": cfg.get("MODEL", {}).get("FUSE", {}).get("MAX_BATCH_TOKENS", 49152),
        "chunk_size": cfg.get("MODEL", {}).get("FUSE", {}).get("CHUNK_SIZE", 64),
        "chunk_shape": cfg.get("MODEL", {}).get("FUSE", {}).get("CHUNK_SHAPE", "square"),
    }
    
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
    ) if use_spike else vrt
    
    # Load checkpoint if specified or from exp_dir
    if args.exp_dir:
        print(f"[test.py] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[test.py] Warning: Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(f"[test.py] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        print(f"[test.py] Checkpoint loaded successfully!")
    else:
        # Old checkpoint finding logic
        checkpoint_path = cfg.get("TEST", {}).get("CHECKPOINT_PATH") or cfg.get("TRAIN", {}).get("CHECKPOINT_PATH")
        if checkpoint_path is None:
            # Auto-find the latest checkpoint in outputs/ckpts/
            ckpt_dir = Path(cfg.get("LOG", {}).get("SAVE_DIR", "outputs")) / "ckpts"
            if ckpt_dir.exists():
                ckpt_files = list(ckpt_dir.glob("*.pth"))
                if ckpt_files:
                    # Use 'last.pth' if it exists, otherwise use the most recent file
                    last_ckpt = ckpt_dir / "last.pth"
                    if last_ckpt.exists():
                        checkpoint_path = str(last_ckpt)
                    else:
                        checkpoint_path = str(max(ckpt_files, key=lambda p: p.stat().st_mtime))
        
        if checkpoint_path:
            print(f"[test.py] Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Handle different checkpoint formats
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            # Load state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[test.py] Warning: Missing keys in checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"[test.py] Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print(f"[test.py] Checkpoint loaded successfully!")
        else:
            print("[test.py] WARNING: No checkpoint found! Using randomly initialized weights.")
    
    model = model.to(device)
    model.eval()

    if args.exp_dir:
        out_dir = exp_dir / "test_results" / "visuals" if not use_dir_api else Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = exp_dir / "test_results" / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(cfg["LOG"].get("SAVE_DIR", "outputs")) / "visuals" if not use_dir_api else Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir = Path(cfg["LOG"].get("SAVE_DIR", "outputs")) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    t0 = time.time()
    num_frames = 0
    with torch.no_grad():
        if not use_dir_api:
            for i, batch in enumerate(val_loader):
                blur = batch["blur"].to(device)
                spike_vox = batch["spike_vox"].to(device)
                
                # Use tile-based inference for large images
                print(f"Processing batch {i}, blur shape: {blur.shape}")
                recon = tile_inference(
                    model, 
                    blur, 
                    spike_vox, 
                    tile_size=args.tile_size, 
                    tile_overlap=args.tile_overlap,
                    device=device,
                    force_tiling=args.enable_tiling,
                    disable_tiling=args.disable_tiling
                )
                
                recon = torch.clamp(recon, 0.0, 1.0)
                import torchvision.utils as vutils
                b0 = recon[0]
                for t in range(b0.shape[0]):
                    vutils.save_image(b0[t], str(out_dir / f"val_{i:04d}_{t:02d}.png"))
                num_frames += recon.shape[0] * recon.shape[1]
                
                # Clear cache to prevent memory accumulation
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        else:
            import numpy as np
            import torchvision.transforms.functional as TF
            import torchvision.utils as vutils
            # 简化：单帧推理，按 clip_len=1 输出
            imgs = sorted([p for p in blur_dir.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}])
            for i, img_path in enumerate(imgs):
                stem = img_path.stem
                vox_path = vox_dir / f"{stem}.npy"
                if not vox_path.exists():
                    continue
                from PIL import Image
                im = torch.from_numpy(np.array(Image.open(img_path).convert('RGB'), dtype=np.float32)/255.0).permute(2,0,1)
                KHW = torch.from_numpy(np.load(vox_path).astype('float32'))  # (K,H,W)
                H, W = im.shape[-2:]
                if KHW.shape[-2:] != (H,W):
                    # 仅居中裁剪到最小公共尺寸
                    h = min(H, KHW.shape[-2])
                    w = min(W, KHW.shape[-1])
                    im = TF.center_crop(im, [h,w])
                    KHW = KHW[:, :h, :w]
                blur = im.unsqueeze(0).unsqueeze(0)   # (1,1,3,H,W)
                vox = KHW.unsqueeze(0).unsqueeze(0)   # (1,1,K,H,W)
                recon = model(blur.to(device), vox.to(device))
                recon = torch.clamp(recon, 0.0, 1.0)
                vutils.save_image(recon[0,0], str(out_dir / f"{stem}.png"))
                num_frames += 1
    dt = time.time() - t0
    fps = num_frames / max(dt, 1e-6)
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0.0
    with open(metrics_dir / "test_summary.json", 'w', encoding='utf-8') as f:
        json.dump({"frames": int(num_frames), "seconds": dt, "fps": fps, "peak_mem_mb": peak_mem}, f, ensure_ascii=False, indent=2)
    print("[test] Done: ", {"fps": fps, "peak_mem_mb": peak_mem})


if __name__ == "__main__":
    main()



