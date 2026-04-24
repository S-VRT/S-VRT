from pathlib import Path
import os

import cv2
import numpy as np
import torch

from utils import utils_image as util


class FusionDebugDumper:
    """Centralized, opt-in fusion output capture and image dumping."""

    def __init__(self, opt):
        self.opt = opt
        fusion_cfg = opt.get('netG', {}).get('fusion', {})
        raw_cfg = fusion_cfg.get('debug', {}) if isinstance(fusion_cfg.get('debug', {}), dict) else {}
        self.cfg = raw_cfg
        self.placement = str(fusion_cfg.get('placement', 'early')).strip().lower()
        self.enabled = bool(raw_cfg.get('enable', raw_cfg.get('save_images', False)))
        self.save_images = bool(raw_cfg.get('save_images', self.enabled))
        self.trigger = str(raw_cfg.get('trigger', 'phase1_last')).strip().lower()
        self.source = str(raw_cfg.get('source', 'train_batch')).strip().lower()
        self.source_view = str(raw_cfg.get('source_view', 'main')).strip().lower()
        self.subdir = str(raw_cfg.get('subdir', 'fusion_phase1_last'))
        self.max_batches = max(1, int(raw_cfg.get('max_batches', 1)))
        self.max_frames = raw_cfg.get('max_frames', None)
        if self.max_frames is not None:
            self.max_frames = max(1, int(self.max_frames))
        self.full_frame = bool(raw_cfg.get('full_frame', False))
        self.val_overrides = raw_cfg.get('val_overrides', {}) if isinstance(raw_cfg.get('val_overrides', {}), dict) else {}
        self._armed = False
        self._last_output = None
        self._hooks = []

    def attach(self, model):
        self.close()
        if not self.enabled:
            return
        target = getattr(model, 'fusion_adapter', None)
        if target is None:
            return
        self._hooks.append(target.register_forward_hook(self._capture_hook))

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def arm(self):
        if self.enabled:
            self._last_output = None
            self._armed = True

    def disarm(self):
        self._armed = False

    def _capture_hook(self, module, inputs, output):
        if not self._armed:
            return
        tensor = None
        if isinstance(output, dict):
            tensor = output.get("fused_main", None)
        elif isinstance(output, (tuple, list)):
            tensor = output[0]
        else:
            tensor = output
        if isinstance(tensor, torch.Tensor):
            self.capture_tensor(tensor)

    def capture_tensor(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        tensor = self._normalize_layout(tensor)
        self._last_output = tensor.detach().cpu()

    def _normalize_layout(self, tensor):
        if tensor.ndim != 5:
            return tensor
        if self.placement == 'middle':
            return tensor.permute(0, 2, 1, 3, 4)
        if self.placement == 'hybrid' and tensor.size(2) > 4 and tensor.size(1) <= 4:
            return tensor.permute(0, 2, 1, 3, 4)
        return tensor

    def should_dump_phase1_last(self, current_step, fix_iter, source=None):
        if not (self.enabled and self.save_images):
            return False
        if self.trigger != 'phase1_last':
            return False
        if fix_iter <= 0 or current_step != fix_iter - 1:
            return False
        if source is not None and self.source != source:
            return False
        return True

    def _dump_selected_tensor(self, fusion, current_step, folder, lq_paths, gt=None, spike_bins=None, rank=0):
        if rank != 0 or not (self.enabled and self.save_images):
            return False
        if fusion is None or fusion.ndim != 5 or fusion.size(2) < 3:
            return False
        folder_name = self._resolve_folder_name(folder)
        save_root = Path(self.opt['path']['images']) / folder_name / self.subdir
        save_root.mkdir(parents=True, exist_ok=True)

        frames = fusion.size(1)
        if self.max_frames is not None:
            frames = min(frames, self.max_frames)
        metric_rows = []
        for i in range(fusion.size(0)):
            clip_name = self._resolve_clip_name(lq_paths, i)
            frame_paths = self._resolve_frame_paths(lq_paths, i)
            rgb_frames = len(frame_paths)
            clip_spike_bins = spike_bins
            if clip_spike_bins is None and rgb_frames > 0 and fusion.size(1) % rgb_frames == 0:
                clip_spike_bins = fusion.size(1) // rgb_frames
            clip = fusion[i].float().clamp_(0, 1).numpy()
            gt_clip = self._select_gt_clip(gt, i)
            if gt_clip is not None:
                metric_rows.extend(
                    self._calculate_metric_rows(
                        clip_name=clip_name,
                        clip=clip,
                        gt_clip=gt_clip,
                        spike_bins=clip_spike_bins,
                    )
                )
            for t in range(frames):
                img = clip[t, :3, :, :]
                img = self._chw_rgb_float_to_bgr_uint(img)
                if (
                    clip_spike_bins is not None
                    and clip_spike_bins > 0
                    and rgb_frames > 0
                    and clip.shape[0] == rgb_frames * clip_spike_bins
                ):
                    rgb_idx = t // clip_spike_bins
                    spike_idx = t % clip_spike_bins
                    filename = (
                        f'{clip_name}_fusion_rgb{rgb_idx:03d}_spk{spike_idx:02d}_t{t:03d}_{current_step:d}.png'
                    )
                else:
                    filename = f'{clip_name}_fusion_t{t:03d}_{current_step:d}.png'
                cv2.imwrite(str(save_root / filename), img)
        if metric_rows:
            self._write_metric_rows(save_root / f'fusion_metrics_{current_step:d}.csv', metric_rows)
        return True

    def dump_tensor(
        self,
        fusion_main,
        current_step,
        folder,
        gt=None,
        rank=0,
        lq_paths=None,
        fusion_exec=None,
        fusion_meta=None,
        source_view=None,
    ):
        selected_source = str(source_view or self.source_view or "main").strip().lower()
        if selected_source not in {"main", "exec"}:
            raise ValueError(f"Unsupported fusion debug source_view={selected_source!r}.")

        fusion = fusion_exec if selected_source == "exec" else fusion_main
        if not isinstance(fusion, torch.Tensor):
            return False

        spike_bins = None
        if selected_source == "exec" and isinstance(fusion_meta, dict):
            raw_spike_bins = fusion_meta.get("spike_bins", None)
            if raw_spike_bins is not None:
                spike_bins = int(raw_spike_bins)

        return self._dump_selected_tensor(
            fusion=fusion.detach().cpu(),
            current_step=current_step,
            folder=folder,
            lq_paths=lq_paths,
            gt=gt,
            spike_bins=spike_bins,
            rank=rank,
        )

    def dump_last(self, current_step, folder, lq_paths, gt=None, spike_bins=None, rank=0):
        fusion = self._last_output
        self._last_output = None
        return self._dump_selected_tensor(
            fusion=fusion,
            current_step=current_step,
            folder=folder,
            lq_paths=lq_paths,
            gt=gt,
            spike_bins=spike_bins,
            rank=rank,
        )

    def _calculate_metric_rows(self, clip_name, clip, gt_clip, spike_bins=None):
        if gt_clip.ndim != 4 or gt_clip.shape[1] < 3:
            return []
        rows = []
        gt_frames = gt_clip.shape[0]
        if spike_bins is not None and spike_bins > 0 and clip.shape[0] == gt_frames * spike_bins:
            frame_indices = range(gt_frames)
            time_indices = [idx * spike_bins + spike_bins // 2 for idx in frame_indices]
        elif clip.shape[0] == gt_frames:
            frame_indices = range(gt_frames)
            time_indices = list(frame_indices)
        else:
            return []

        for frame_idx, time_idx in zip(frame_indices, time_indices):
            if time_idx >= clip.shape[0]:
                continue
            pred_img = self._chw_rgb_float_to_bgr_uint(clip[time_idx, :3, :, :])
            gt_img = self._chw_rgb_float_to_bgr_uint(gt_clip[frame_idx, :3, :, :])
            psnr = util.calculate_psnr(pred_img, gt_img, border=0)
            ssim = util.calculate_ssim(pred_img, gt_img, border=0)
            if pred_img.ndim == 3:
                pred_y = util.bgr2ycbcr(pred_img.astype(np.float32) / 255.) * 255.
                gt_y = util.bgr2ycbcr(gt_img.astype(np.float32) / 255.) * 255.
                psnr_y = util.calculate_psnr(pred_y, gt_y, border=0)
                ssim_y = util.calculate_ssim(pred_y, gt_y, border=0)
            else:
                psnr_y = psnr
                ssim_y = ssim
            rows.append(
                {
                    'clip': clip_name,
                    'frame': frame_idx,
                    'subframe': time_idx - frame_idx * spike_bins if spike_bins else 0,
                    't': time_idx,
                    'psnr': psnr,
                    'ssim': ssim,
                    'psnr_y': psnr_y,
                    'ssim_y': ssim_y,
                }
            )
        return rows

    @staticmethod
    def _write_metric_rows(path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('clip,frame,subframe,t,psnr,ssim,psnr_y,ssim_y\n')
            for row in rows:
                f.write(
                    f"{row['clip']},{row['frame']},{row['subframe']},{row['t']},"
                    f"{FusionDebugDumper._format_metric(row['psnr'])},"
                    f"{FusionDebugDumper._format_metric(row['ssim'])},"
                    f"{FusionDebugDumper._format_metric(row['psnr_y'])},"
                    f"{FusionDebugDumper._format_metric(row['ssim_y'])}\n"
                )

    @staticmethod
    def _format_metric(value):
        if np.isinf(value):
            return 'inf'
        if np.isnan(value):
            return 'nan'
        return f'{value:.6f}'

    @staticmethod
    def _select_gt_clip(gt, index):
        if gt is None:
            return None
        if isinstance(gt, torch.Tensor):
            if gt.ndim == 5 and index < gt.size(0):
                return gt[index].detach().float().clamp(0, 1).cpu().numpy()
            if gt.ndim == 4 and index == 0:
                return gt.detach().float().clamp(0, 1).cpu().numpy()
        return None

    @staticmethod
    def _chw_rgb_float_to_bgr_uint(img):
        img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
        return (img * 255.0).round().astype(np.uint8)

    @staticmethod
    def _resolve_folder_name(folder):
        if isinstance(folder, (list, tuple)) and folder:
            folder = folder[0]
        return str(folder) if folder is not None else 'unknown'

    @staticmethod
    def _resolve_clip_name(lq_paths, index):
        frame_paths = FusionDebugDumper._resolve_frame_paths(lq_paths, index)
        if frame_paths:
            clip_source = frame_paths[0]
            if isinstance(clip_source, str):
                return os.path.splitext(os.path.basename(clip_source))[0]
        if lq_paths is not None:
            clip_source = None
            try:
                clip_source = lq_paths[index]
            except Exception:
                clip_source = None
            if isinstance(clip_source, (list, tuple)) and clip_source:
                clip_source = clip_source[0]
            if isinstance(clip_source, str):
                return os.path.splitext(os.path.basename(clip_source))[0]
        return f'clip_{index:03d}'

    @staticmethod
    def _resolve_frame_paths(lq_paths, index):
        if lq_paths is None:
            return []
        try:
            sample_paths = lq_paths[index]
        except Exception:
            sample_paths = lq_paths
        if isinstance(sample_paths, (list, tuple)):
            return [path for path in sample_paths if isinstance(path, str)]
        if isinstance(sample_paths, str):
            return [sample_paths]
        return []


__all__ = ['FusionDebugDumper']
