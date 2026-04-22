from pathlib import Path
import os

import cv2
import numpy as np
import torch


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
        tensor = output[0] if isinstance(output, (tuple, list)) else output
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

    def dump_last(self, current_step, folder, lq_paths, rank=0):
        if rank != 0 or not (self.enabled and self.save_images):
            return False
        fusion = self._last_output
        self._last_output = None
        if fusion is None or fusion.ndim != 5 or fusion.size(2) < 3:
            return False

        folder_name = self._resolve_folder_name(folder)
        save_root = Path(self.opt['path']['images']) / folder_name / self.subdir
        save_root.mkdir(parents=True, exist_ok=True)

        frames = fusion.size(1)
        if self.max_frames is not None:
            frames = min(frames, self.max_frames)
        for i in range(fusion.size(0)):
            clip_name = self._resolve_clip_name(lq_paths, i)
            frame_paths = self._resolve_frame_paths(lq_paths, i)
            rgb_frames = len(frame_paths)
            spike_bins = None
            if rgb_frames > 0 and fusion.size(1) % rgb_frames == 0:
                spike_bins = fusion.size(1) // rgb_frames
            clip = fusion[i].float().clamp_(0, 1).numpy()
            for t in range(frames):
                img = clip[t, :3, :, :]
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
                img = (img * 255.0).round().astype(np.uint8)
                if spike_bins is not None and spike_bins > 0:
                    rgb_idx = t // spike_bins
                    spike_idx = t % spike_bins
                    filename = (
                        f'{clip_name}_fusion_rgb{rgb_idx:03d}_spk{spike_idx:02d}_t{t:03d}_{current_step:d}.png'
                    )
                else:
                    filename = f'{clip_name}_fusion_t{t:03d}_{current_step:d}.png'
                cv2.imwrite(str(save_root / filename), img)
        return True

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
