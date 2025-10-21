import argparse
import os
import sys
from pathlib import Path
import json
import shutil
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare spike voxel caches and dataset statistics for VRT+Spike baseline"
    )
    parser.add_argument("--data-root", type=str, required=True, help="DATA_ROOT directory (raw/x4k1000fps)")
    parser.add_argument("--fps", type=int, default=1000, help="Source FPS for spike/blurry alignment")
    parser.add_argument("--exposure-frames", type=int, default=33, help="Exposure frames e for blurry synthesis")
    parser.add_argument("--K", type=int, default=32, help="Number of temporal bins for voxelization")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to process")
    parser.add_argument("--out-npy", action="store_true", help="Persist voxel grids to .npy")
    parser.add_argument("--voxel-dirname", type=str, default="spike_vox", help="Directory name for cached voxels under each sequence")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing voxel caches")
    parser.add_argument("--dry-run", action="store_true", help="List targets without writing")
    parser.add_argument("--align-log", type=str, default=None, help="Path to write alignment log; defaults under outputs/logs")
    parser.add_argument("--config", type=str, default="configs/deblur/vrt_spike_baseline.yaml", help="Config yaml to update NORM stats")
    return parser.parse_args()


def find_sequences(root: Path) -> List[Path]:
    seqs = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for seq in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            seqs.append(seq)
    return seqs


def compute_voxel_stats(voxel_paths: List[Path]) -> Dict[str, float]:
    count = 0
    mean_acc = 0.0
    m2_acc = 0.0
    for p in voxel_paths:
        try:
            arr = np.load(p)
        except Exception:
            continue
        x = arr.astype(np.float64)
        n = x.size
        if n == 0:
            continue
        x_mean = float(x.mean())
        x_var = float(x.var())
        # Welford combine
        total = count + n
        delta = x_mean - (mean_acc if count > 0 else 0.0)
        mean_acc = (count * mean_acc + n * x_mean) / total
        m2_acc = m2_acc + n * x_var + (delta * delta) * (count * n) / total
        count = total
    std = float(np.sqrt(m2_acc / max(count, 1))) if count > 0 else 1.0
    return {"mean": float(mean_acc), "std": std}


def write_align_log(log_path: Path, records: List[Tuple[str, int, float, float, int]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("frame_idx,t0,t1,event_count,seq\n")
        for seq, idx, t0, t1, cnt in records:
            f.write(f"{idx},{t0:.6f},{t1:.6f},{cnt},{seq}\n")


def update_config_norm(config_path: Path, mean: float, std: float) -> None:
    try:
        import yaml
    except Exception:
        print("[prepare_data] PyYAML not available; skip config update.")
        return
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "DATA" in cfg and "NORM" in cfg["DATA"]:
        cfg["DATA"]["NORM"]["MEAN"] = float(mean)
        cfg["DATA"]["NORM"]["STD"] = float(std)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    # 确保项目根目录可导入（src/*）
    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    print("[prepare_data]")
    print(f"- data_root           : {data_root}")
    print(f"- fps                 : {args.fps}")
    print(f"- exposure_frames     : {args.exposure_frames}")
    print(f"- bins (K)            : {args.K}")
    print(f"- split               : {args.split}")
    print(f"- voxel cache dirname : {args.voxel_dirname}")
    print(f"- overwrite           : {args.overwrite}")
    print(f"- dry_run             : {args.dry_run}")

    # Phase 1: locate existing voxel caches
    sequences = find_sequences(data_root)
    voxel_files: List[Path] = []
    align_records: List[Tuple[str, int, float, float, int]] = []
    for seq in sequences:
        vox_dir = seq / args.voxel_dirname
        if vox_dir.exists():
            voxel_files.extend(sorted(vox_dir.glob("*.npy")))
        # placeholder: if raw spike exists, here we should generate voxels and record alignment
        # Without raw spec, we log a dummy alignment with unknown t0/t1
        blur_dir = seq / "blur"
        if blur_dir.exists():
            for img in sorted(blur_dir.glob("*.png")):
                stem = img.stem
                # Unknown timestamps; set 0.0 placeholders and count from existing vox if available
                vox_p = vox_dir / f"{stem}.npy"
                cnt = int(np.load(vox_p).sum()) if vox_p.exists() else 0
                align_records.append((seq.name, int(stem), 0.0, 0.0, cnt))

    print(f"- found sequences     : {len(sequences)}")
    print(f"- found voxel files   : {len(voxel_files)}")

    if args.dry_run:
        for p in voxel_files[:10]:
            print(f"  sample voxel: {p}")
        print("[prepare_data] dry-run complete.")
        return

    # Phase 2: generate voxels if missing (requires vendor adapter)
    from src.data.vendors.x4k.io_x4k import (
        load_spike_sequence,
        list_blur_frames_for_scene,
        spike_to_blur_spatial_align,
        parse_frame_index_from_name,
    )
    used_split = args.split
    blurry_root = data_root / f"{used_split}_blurry_33"
    spike_root = data_root / f"{used_split}_spike_2xds"
    # If requested split not present, fallback to the other one (prefer val)
    if not blurry_root.exists() or not spike_root.exists():
        alt = "val" if args.split != "val" else "train"
        alt_blur = data_root / f"{alt}_blurry_33"
        alt_spk = data_root / f"{alt}_spike_2xds"
        if alt_blur.exists() and alt_spk.exists():
            used_split = alt
            blurry_root = alt_blur
            spike_root = alt_spk
            print(f"[prepare_data] fallback split -> {used_split}")
        else:
            print("[prepare_data] No matching split dirs found under data_root; skip voxel generation.")
    processed_root = Path("data/processed/x4k1000fps")
    out_root = processed_root / args.voxel_dirname
    if args.out_npy and blurry_root.exists() and spike_root.exists():
        for scene in sorted(blurry_root.glob("*/*")):
            if not scene.is_dir():
                continue
            rel = scene.relative_to(blurry_root)
            blur_list = list_blur_frames_for_scene(scene)
            if len(blur_list) == 0:
                continue
            spike_dat = spike_root / rel / "spike.dat"
            try:
                spike_seq = load_spike_sequence(spike_dat, None, None)  # (T,Hs,Ws)
            except NotImplementedError:
                print("[prepare_data] Vendor loader not wired; skip voxel generation. Only stats and logs will be produced.")
                break
            # 读取第一帧获取目标尺寸（用 Pillow，避免额外依赖）
            from PIL import Image
            with Image.open(blur_list[0]) as _im:
                H, W = _im.size[1], _im.size[0]
            spike_seq = spike_to_blur_spatial_align(spike_seq, (H, W))
            # 遍历每个 blur 帧，构造 t0,t1（以帧为单位）
            scene_out = out_root / rel
            scene_out.mkdir(parents=True, exist_ok=True)
            for img in blur_list:
                # 兼容 0004.png / 0007_002.png 等命名
                stem_name = img.stem
                frame_idx = int(parse_frame_index_from_name(img))
                e = int(args.exposure_frames)
                t0 = frame_idx - (e - 1) / 2.0
                t1 = frame_idx + (e - 1) / 2.0
                # 将 [t0,t1] 均分为 K 段，按“帧号等间隔”计数
                K = int(args.K)
                Ht, Wt = spike_seq.shape[-2:]
                vox = np.zeros((K, Ht, Wt), dtype=np.float32)
                if spike_seq.ndim == 3:
                    T = spike_seq.shape[0]
                    # 近似：将 spike_seq 的帧索引当时间戳
                    ts = np.arange(T, dtype=np.float32)
                    # 统计每个时间帧的总脉冲作为权重，分配到对应 bin
                    # 简化：对每帧整图直接累加到最近的 bin
                    # 更精确的事件级别需 vendor 提供时间戳
                    bin_idx = np.floor((ts - t0) / max((t1 - t0), 1e-9) * K).astype(np.int64)
                    bin_idx = np.clip(bin_idx, 0, K - 1)
                    for ti, bi in enumerate(bin_idx):
                        vox[bi] += spike_seq[ti].astype(np.float32)
                # 归一化前先 log1p
                vox = np.log1p(vox, dtype=np.float32)
                np.save(scene_out / f"{stem_name}.npy", vox)
                voxel_files.append(scene_out / f"{stem_name}.npy")
                align_records.append((f"{used_split}/{str(rel)}", int(frame_idx), float(t0 / args.fps), float(t1 / args.fps), int(vox.sum())))

    # Phase 3: compute stats over existing voxels
    stats = compute_voxel_stats(voxel_files)
    stats_path = data_root / "voxel_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"- saved voxel stats to: {stats_path}")

    # Phase 4: write align log
    log_path = Path(args.align_log) if args.align_log else (Path("outputs") / "logs" / f"align_{data_root.name}.txt")
    write_align_log(log_path, align_records)
    print(f"- wrote align log     : {log_path}")

    # Phase 5: update config NORM
    update_config_norm(Path(args.config), stats["mean"], stats["std"])
    print(f"- updated config NORM : {args.config}")

    # Phase 6: Build unified dataset root for loader (Plan A)
    # Structure: data/processed/x4k1000fps_unified/{used_split}/{seq}/{blur,sharp,spike_vox}
    unify_root = Path("data/processed/x4k1000fps_unified")
    if blurry_root.exists():
        for scene in sorted(blurry_root.glob("*/*")):
            if not scene.is_dir():
                continue
            rel = scene.relative_to(blurry_root)
            dst_seq_root = unify_root / used_split / rel
            dst_blur = dst_seq_root / "blur"
            dst_sharp = dst_seq_root / "sharp"
            dst_vox = dst_seq_root / "spike_vox"
            # ensure dirs
            dst_blur.mkdir(parents=True, exist_ok=True)
            dst_sharp.mkdir(parents=True, exist_ok=True)
            dst_vox.mkdir(parents=True, exist_ok=True)
            # copy blur images
            for img in sorted(scene.glob("*.png")):
                shutil.copy2(img, dst_blur / img.name)
            # copy spike_vox npy if produced
            src_vox_dir = out_root / rel if out_root.exists() else None
            if src_vox_dir and src_vox_dir.exists():
                for npy in sorted(src_vox_dir.glob("*.npy")):
                    shutil.copy2(npy, dst_vox / npy.name)
            # placeholder sharp if no GT
            # (dev stage only; if真实sharp存在可替换此目录)
            for img in sorted(dst_blur.glob("*.png")):
                sharp_dst = dst_sharp / img.name
                if not sharp_dst.exists():
                    shutil.copy2(img, sharp_dst)
        print(f"- built unified root  : {unify_root} (for loader ROOT)")


if __name__ == "__main__":
    main()


