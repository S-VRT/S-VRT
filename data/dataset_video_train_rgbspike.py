import numpy as np
import os
import random
import torch
from pathlib import Path
import torch.utils.data as data
from torch.utils.data import get_worker_info
import cv2

import utils.utils_video as utils_video
from data.spike_recc import SpikeStream, extract_centered_raw_window, voxelize_spikes_tfp
from data.spike_recc.middle_tfp.reconstructor import MiddleTFPReconstructor
from data.spike_recc.encoding25 import (
    load_encoding25_artifact_with_shape,
    validate_encoding25_tensor,
    validate_subframes_tensor,
)


def resize_chw_image(arr_chw, size):
    """Resize a CHW tensor-like image through OpenCV's HWC multi-channel path."""
    arr_hwc = np.transpose(arr_chw, (1, 2, 0))
    resized_hwc = cv2.resize(arr_hwc, size, interpolation=cv2.INTER_LINEAR)
    if resized_hwc.ndim == 2:
        resized_hwc = resized_hwc[:, :, np.newaxis]
    return np.transpose(resized_hwc, (2, 0, 1)).astype(np.float32)


class TrainDatasetRGBSpike(data.Dataset):
    """Video dataset for training recurrent networks with RGB + Spike data.

    This dataset extends TrainDataset to support loading both
    RGB frames and corresponding Spike camera data, with configurable
    input packing:
      - concat: return combined tensor in key `L`
      - dual: return split tensors in keys `L_rgb` and `L_spike`
        and optionally keep legacy `L` when `keep_legacy_l=True`

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_XXX_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape; 4. start frame, separated by
    a white space.
    Examples:
    720p_240fps_1 100 (720,1280,3) 0
    720p_240fps_3 100 (720,1280,3) 0
    ...

    Key examples: "720p_240fps_1/00000"
    GT (gt): Ground-Truth RGB frames;
    LQ (lq): Low-Quality RGB frames, e.g., low-resolution/blurry/noisy/compressed frames.
    Spike: Spike camera data (.dat files).


    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_spike (str): Data root path for spike data.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            
            spike_h (int): Spike camera height. Default: 250.
            spike_w (int): Spike camera width. Default: 400.
            spike_channels (int): Legacy alias for spike reconstruction bins.
                Prefer spike.reconstruction.num_bins when configuring new experiments.
            input_pack_mode (str): Input packing mode. One of ['concat', 'dual'].
                Default: 'concat'.
            keep_legacy_l (bool): Legacy alias for compat.keep_legacy_L.
                Default: True.
            spike_flipud (bool): Whether to flip spike data vertically. Default: True.
            tfp_half_win_length (int): Half window length fed into TFP. Default: 20.
            tfp_device (str): Torch device string for TFP reconstruction. Default: 'cpu'.

    """

    def __init__(self, opt):
        super(TrainDatasetRGBSpike, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.spike_root = Path(opt['dataroot_spike'])
        self.filename_tmpl = opt.get('filename_tmpl', '08d')
        self._parse_spike_flow_config(opt, optical_flow_module=self._flow_module_name_from_opt(opt))
        self.filename_ext = opt.get('filename_ext', 'png')
        self.num_frame = opt['num_frame']

        # Spike configuration
        self.spike_h = opt.get('spike_h', 360)
        self.spike_w = opt.get('spike_w', 640)
        spike_cfg = opt.get('spike', {}) if isinstance(opt.get('spike', {}), dict) else {}
        spike_precomputed_cfg = spike_cfg.get('precomputed', {}) if isinstance(spike_cfg.get('precomputed', {}), dict) else {}
        compat_cfg = opt.get('compat', {}) if isinstance(opt.get('compat', {}), dict) else {}
        spike_reconstruction_nested = spike_cfg.get('reconstruction', {})
        if not isinstance(spike_reconstruction_nested, dict):
            spike_reconstruction_nested = {}
        nested_num_bins = spike_reconstruction_nested.get('num_bins', None)
        legacy_reconstruction = opt.get('spike_reconstruction', None)
        legacy_reconstruction_cfg = legacy_reconstruction if isinstance(legacy_reconstruction, dict) else {}
        if spike_reconstruction_nested and legacy_reconstruction is not None:
            nested_type = str(spike_reconstruction_nested.get('type', 'spikecv_tfp')).strip().lower()
            if legacy_reconstruction_cfg:
                legacy_type = str(legacy_reconstruction_cfg.get('type', 'spikecv_tfp')).strip().lower()
            else:
                legacy_type = str(legacy_reconstruction).strip().lower()
            if nested_type != legacy_type:
                raise ValueError(
                    f"[TrainDatasetRGBSpike] Conflicting reconstruction types: "
                    f"spike.reconstruction.type={nested_type!r} vs spike_reconstruction={legacy_type!r}."
                )
        legacy_middle_center = opt.get('middle_tfp_center', None)
        if legacy_middle_center is None and legacy_reconstruction_cfg and 'middle_tfp_center' in legacy_reconstruction_cfg:
            legacy_middle_center = legacy_reconstruction_cfg.get('middle_tfp_center')
        if spike_reconstruction_nested and legacy_middle_center is not None and 'middle_tfp_center' in spike_reconstruction_nested:
            nested_center = int(spike_reconstruction_nested['middle_tfp_center'])
            if nested_center != int(legacy_middle_center):
                raise ValueError(
                    f"[TrainDatasetRGBSpike] Conflicting middle_tfp_center values: "
                    f"spike.reconstruction.middle_tfp_center={nested_center} vs "
                    f"middle_tfp_center={int(legacy_middle_center)}."
                )
        self.spike_representation = str(spike_cfg.get('representation', 'tfp')).strip().lower()
        if self.spike_representation not in {'tfp', 'raw_window'}:
            raise ValueError(
                f"[TrainDatasetRGBSpike] spike.representation must be one of {{'tfp', 'raw_window'}}, "
                f"got {self.spike_representation!r}."
            )

        if self.spike_representation == 'raw_window':
            raw_window_length_cfg = spike_cfg.get('raw_window_length', None)
            if raw_window_length_cfg is None:
                raw_window_length_cfg = 2 * int(opt.get('tfp_half_win_length', 20)) + 1
            self.raw_window_length = int(raw_window_length_cfg)
            if self.raw_window_length <= 0 or self.raw_window_length % 2 == 0:
                raise ValueError(
                    f"[TrainDatasetRGBSpike] raw_window_length must be a positive odd integer, "
                    f"got {self.raw_window_length}."
                )
        else:
            self.raw_window_length = None

        if self.spike_representation == 'raw_window':
            default_spike_channels = self.raw_window_length
        else:
            default_spike_channels = int(nested_num_bins) if nested_num_bins is not None else 4

        self.spike_channels = int(opt.get('spike_channels', default_spike_channels))
        if self.spike_representation == 'raw_window':
            if 'spike_channels' in opt and int(opt['spike_channels']) != self.raw_window_length:
                raise ValueError(
                    f"[TrainDatasetRGBSpike] Conflicting channel settings for raw_window: "
                    f"spike_channels={int(opt['spike_channels'])} vs raw_window_length={self.raw_window_length}."
                )
        elif nested_num_bins is not None and 'spike_channels' in opt and int(opt['spike_channels']) != int(nested_num_bins):
            raise ValueError(
                f"[TrainDatasetRGBSpike] Conflicting channel settings: spike_channels={int(opt['spike_channels'])} "
                f"vs spike.reconstruction.num_bins={int(nested_num_bins)}."
            )
        if self.use_encoding25_flow and self.spike_flow_subframes > 1:
            if self.spike_flow_subframes != self.spike_channels:
                raise ValueError(
                    f"spike_flow.subframes ({self.spike_flow_subframes}) must equal "
                    f"spike_channels ({self.spike_channels}) for early-fusion temporal "
                    f"axis alignment."
                )
        self.spike_flipud = opt.get('spike_flipud', True)
        self.tfp_half_win_length = opt.get('tfp_half_win_length', 20)
        self.use_precomputed_spike = bool(spike_precomputed_cfg.get('enable', False))
        self.precomputed_spike_format = str(spike_precomputed_cfg.get('format', 'npy')).strip().lower()
        precomputed_root_value = spike_precomputed_cfg.get('root', 'auto')
        if str(precomputed_root_value).strip().lower() == 'auto':
            self.precomputed_spike_root = self.spike_root
        else:
            self.precomputed_spike_root = Path(precomputed_root_value)
        spike_reconstruction_cfg = spike_reconstruction_nested or opt.get('spike_reconstruction', 'spikecv_tfp')
        if isinstance(spike_reconstruction_cfg, dict):
            self.spike_reconstruction = str(spike_reconstruction_cfg.get('type', 'spikecv_tfp')).strip().lower()
            self.middle_tfp_center = int(
                spike_reconstruction_cfg.get('middle_tfp_center', opt.get('middle_tfp_center', 44))
            )
        else:
            self.spike_reconstruction = str(spike_reconstruction_cfg).strip().lower()
            self.middle_tfp_center = int(opt.get('middle_tfp_center', 44))

        if self.spike_representation == 'raw_window':
            if self.spike_reconstruction != 'spikecv_tfp':
                raise ValueError(
                    "[TrainDatasetRGBSpike] raw_window representation requires spike.reconstruction.type='spikecv_tfp'."
                )

        self._middle_tfp_reconstructor = None
        self._snn_reconstructor = None

        # Initialize TFP device settings before using them
        self._tfp_device_pool = self._normalize_tfp_device_pool(opt.get('tfp_devices', None))
        self.tfp_device = self._normalize_single_tfp_device(opt.get('tfp_device', 'cpu'))
        if self._tfp_device_pool:
            # Multi-device mode takes precedence over single-device mode.
            self.tfp_device = None

        # Initialize spike reconstructors
        if self.spike_reconstruction in {'middle_tfp', 'middle-tfp'}:
            if self.spike_representation == 'raw_window':
                raise ValueError(
                    "[TrainDatasetRGBSpike] raw_window representation requires spike.reconstruction.type='spikecv_tfp'."
                )
            if self.spike_channels != 1:
                self.spike_channels = 1
            self._middle_tfp_reconstructor = MiddleTFPReconstructor(
                spike_h=self.spike_h,
                spike_w=self.spike_w,
                center=self.middle_tfp_center,
            )
        elif self.spike_reconstruction == 'snn':
            if self.spike_representation == 'raw_window':
                raise ValueError(
                    "[TrainDatasetRGBSpike] raw_window representation requires spike.reconstruction.type='spikecv_tfp'."
                )
            if self.spike_channels != 1:
                self.spike_channels = 1
            snn_cfg = spike_reconstruction_cfg if isinstance(spike_reconstruction_cfg, dict) else {}
            try:
                from data.spike_recc.snn.reconstructor import SNNReconstructor
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "[TrainDatasetRGBSpike] spike_reconstruction='snn' requires optional dependency "
                    "'snntorch'. Install it before using SNN reconstruction."
                ) from exc
            # Use device from config or fallback to tfp_device
            snn_device = snn_cfg.get('device', self.tfp_device if self.tfp_device else 'cpu')
            self._snn_reconstructor = SNNReconstructor(
                checkpoint_path=snn_cfg.get('checkpoint_path', 'checkpoints/snn_epoch_100.pth'),
                spike_win=snn_cfg.get('spike_win', 8),
                center=self.middle_tfp_center,
                device=snn_device
            )
        self.rgb_norm_stats = self._build_norm_stats(opt.get('rgb_normalize', None), num_channels=3, preset='imagenet')
        self.spike_norm_stats = self._build_norm_stats(opt.get('spike_normalize', None), num_channels=self.spike_channels)
        raw_input_pack_mode = opt.get('input_pack_mode', 'concat')
        if raw_input_pack_mode is None:
            raw_input_pack_mode = 'concat'
        normalized_input_pack_mode = str(raw_input_pack_mode).strip().lower()
        supported_pack_modes = {'concat', 'dual'}
        if normalized_input_pack_mode not in supported_pack_modes:
            raise ValueError(
                f"[TrainDatasetRGBSpike] input_pack_mode must be one of {supported_pack_modes}, got '{raw_input_pack_mode}'."
            )
        self.input_pack_mode = normalized_input_pack_mode
        self.keep_legacy_l = bool(compat_cfg.get('keep_legacy_L', opt.get('keep_legacy_l', True)))
        self._dual_mode = self.input_pack_mode == 'dual'
        self._precomputed_spike_warned = set()
        # Raw ingress width remains RGB 3 + spike bins; dual mode may still emit the
        # concatenated `L` tensor for compatibility when keep_legacy_l=True.
        self.expected_lq_channels = 3 + self.spike_channels
        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _, start_frame = line.split(' ')
                keys.extend([f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame), int(start_frame)+int(frame_num))])
                total_num_frames.extend([int(frame_num) for i in range(int(frame_num))])
                start_frames.extend([int(start_frame) for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['name'] == 'REDS':
            if opt['val_partition'] == 'REDS4':
                val_partition = ['000', '011', '015', '020']
            elif opt['val_partition'] == 'official':
                val_partition = [f'{v:03d}' for v in range(240, 270)]
            else:
                raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                                 f"Supported ones are ['official', 'REDS4'].")
        else:
            val_partition = []

        self.keys = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames = []
        if opt['test_mode']:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()
        self.io_backend_type = self.io_backend_opt['type']
        self.is_lmdb = False
        if self.io_backend_type == 'lmdb':
            self.is_lmdb = True
            # Auto-add .lmdb suffix if not present
            lq_path = str(self.lq_root)
            gt_path = str(self.gt_root)
            if not lq_path.endswith('.lmdb'):
                lq_path = lq_path + '.lmdb'
            if not gt_path.endswith('.lmdb'):
                gt_path = gt_path + '.lmdb'
            self.io_backend_opt['db_paths'] = [lq_path, gt_path]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
        # FileClient expects the backend type separately
        self.io_backend_opt.pop('type', None)

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')


    def __getitem__(self, index):
        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        spike_voxels = []
        flow_spikes = []
        defer_precomputed_raw_window_crop = (
            self.use_precomputed_spike
            and self.spike_representation == 'raw_window'
            and self.precomputed_spike_format == 'npy'
        )

        img_gt_path_reference = None
        for neighbor in neighbor_list:
            neighbor_key = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            sample = self._load_rgb_frame(neighbor_key) if defer_precomputed_raw_window_crop else self._load_raw_frame(neighbor_key)
            img_lqs.append(sample['lq'])
            img_gts.append(sample['gt'])
            if not defer_precomputed_raw_window_crop:
                spike_voxels.append(sample['spike'])
            if self.use_encoding25_flow:
                flow_spikes.append(self._load_encoded_flow_spike(clip_name, neighbor))
            img_gt_path_reference = sample['gt_path']

        lq_h_orig, lq_w_orig = img_lqs[0].shape[:2]

        # randomly crop RGB frames
        img_gts, img_lqs, crop_params = utils_video.paired_random_crop(
            img_gts, img_lqs, self.gt_size, self.scale, img_gt_path_reference
        )

        cropped_h, cropped_w = img_lqs[0].shape[:2]

        def _crop_resize_chw(arr_chw, expected_channels, name):
            crop = self._resolve_chw_crop(arr_chw.shape, crop_params, lq_h_orig, lq_w_orig, expected_channels, name)
            arr_cropped = self._apply_chw_crop(arr_chw, crop)
            return resize_chw_image(arr_cropped, (cropped_w, cropped_h))

        # Crop spike voxels to the RGB-corresponding region, then resize to the RGB crop size.
        spike_voxels_resized = []
        if defer_precomputed_raw_window_crop:
            spike_shape = (self.spike_channels, self.spike_h, self.spike_w)
            spike_crop = self._resolve_chw_crop(
                spike_shape,
                crop_params,
                lq_h_orig,
                lq_w_orig,
                self.spike_channels,
                "Spike voxel",
            )
            for neighbor in neighbor_list:
                spike_file = self.spike_root / clip_name / 'spike' / f'{neighbor:{self.filename_tmpl}}.dat'
                spike_voxel = self._load_spike_voxel(clip_name, neighbor, spike_file, crop=spike_crop)
                spike_voxels_resized.append(resize_chw_image(spike_voxel, (cropped_w, cropped_h)))
        else:
            for spike_voxel in spike_voxels:
                spike_voxels_resized.append(_crop_resize_chw(spike_voxel, self.spike_channels, "Spike voxel"))

        flow_spikes_resized = []
        if self.use_encoding25_flow:
            for flow_spike in flow_spikes:
                if self.spike_flow_subframes > 1:
                    # Subframe mode: [S, 25, H, W] → crop each sub-window independently
                    for s_idx in range(self.spike_flow_subframes):
                        sub_window = flow_spike[s_idx]  # [25, H, W]
                        validate_encoding25_tensor(sub_window)
                        flow_spikes_resized.append(_crop_resize_chw(sub_window, 25, "Flow spike"))
                else:
                    validate_encoding25_tensor(flow_spike)
                    flow_spikes_resized.append(_crop_resize_chw(flow_spike, 25, "Flow spike"))

        # Concatenate RGB and Spike channels
        # Channel Order: 
        #   0~2: RGB (0-1 float)
        #   3~(3+S-1): Spike voxels (0-1 float, S=self.spike_channels)
        # Total: 3 + self.spike_channels
        img_lqs_with_spike = []
        for img_lq, spike_voxel in zip(img_lqs, spike_voxels_resized):
            # img_lq: (H, W, 3), spike_voxel: (S, H, W)
            # Convert spike_voxel to (H, W, S)
            spike_voxel_hwc = np.transpose(spike_voxel, (1, 2, 0))  # (H, W, S)
            self._validate_raw_rgb_spike_pair(img_lq, spike_voxel_hwc)
            # Concatenate along channel dimension
            img_lq_with_spike = np.concatenate([img_lq, spike_voxel_hwc], axis=2)  # (H, W, 3+S)
            img_lqs_with_spike.append(img_lq_with_spike)

        # augmentation - flip, rotate
        flow_hwc_list = [np.transpose(arr, (1, 2, 0)) for arr in flow_spikes_resized] if self.use_encoding25_flow else []
        merge_list = img_lqs_with_spike + flow_hwc_list + img_gts
        img_results = utils_video.augment(merge_list, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results, bgr2rgb=False)
        lq_count = len(img_lqs_with_spike)
        flow_count = len(flow_hwc_list)
        img_lqs = torch.stack(img_results[:lq_count], dim=0)
        flow_tensor = None
        if flow_count > 0:
            flow_tensor = torch.stack(img_results[lq_count:lq_count + flow_count], dim=0)
        img_gts = torch.stack(img_results[lq_count + flow_count:], dim=0)
        if img_lqs.size(1) != self.expected_lq_channels:
            raise ValueError(
                f"[TrainDatasetRGBSpike] Expected {self.expected_lq_channels} channels "
                f"but received {img_lqs.size(1)} (tensor shape: {img_lqs.shape}). "
                "Double-check spike voxel stacking and augmentation."
            )
        img_lqs = self._apply_channel_normalization(img_lqs)

        sample = {'H': img_gts, 'key': key}
        if self._dual_mode:
            L_rgb = img_lqs[:, :3, :, :]
            L_spike = img_lqs[:, 3:, :, :]
            self._validate_dual_tensor_contract(L_rgb, L_spike)
            sample['L_rgb'] = L_rgb
            sample['L_spike'] = L_spike
            if self.keep_legacy_l:
                sample['L'] = img_lqs
        else:
            sample['L'] = img_lqs

        if self.use_encoding25_flow:
            if flow_tensor is None:
                raise ValueError("SCFlow strict mode expected non-empty flow_tensor.")
            if flow_tensor.ndim != 4 or flow_tensor.size(1) != 25:
                raise ValueError(
                    f"Expected L_flow_spike shape [T*S,25,H,W], got {tuple(flow_tensor.shape)}"
                )
            sample['L_flow_spike'] = flow_tensor.float()
        return sample

    def __len__(self):
        return len(self.keys)

    def _ensure_file_client(self):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_type, **self.io_backend_opt)
        return self.file_client

    def _new_file_client(self):
        """Create an isolated FileClient instance (for threaded cache warmup)."""
        return utils_video.FileClient(self.io_backend_type, **self.io_backend_opt)

    def _resolve_chw_crop(self, arr_shape, crop_params, lq_h_orig, lq_w_orig, expected_channels, name):
        if len(arr_shape) != 3:
            raise ValueError(
                f"[TrainDatasetRGBSpike] {name} must be [C,H,W], got shape {arr_shape}."
            )
        if expected_channels is not None and arr_shape[0] != expected_channels:
            label = "Spike" if name == "Spike voxel" else name
            raise ValueError(
                f"[TrainDatasetRGBSpike] {label} channels mismatch before resize: "
                f"expected {expected_channels}, got {arr_shape[0]}."
            )

        src_h, src_w = arr_shape[1:]
        ratio_h = src_h / lq_h_orig
        ratio_w = src_w / lq_w_orig
        src_crop_h = max(round(crop_params['lq_patch_size'] * ratio_h), 1)
        src_crop_w = max(round(crop_params['lq_patch_size'] * ratio_w), 1)
        src_crop_h = min(src_crop_h, src_h)
        src_crop_w = min(src_crop_w, src_w)
        src_top = round(crop_params['top'] * ratio_h)
        src_left = round(crop_params['left'] * ratio_w)
        src_top = max(min(src_top, src_h - src_crop_h), 0)
        src_left = max(min(src_left, src_w - src_crop_w), 0)
        return {
            'top': src_top,
            'left': src_left,
            'height': src_crop_h,
            'width': src_crop_w,
        }

    @staticmethod
    def _apply_chw_crop(arr_chw, crop):
        return arr_chw[
            :,
            crop['top']:crop['top'] + crop['height'],
            crop['left']:crop['left'] + crop['width'],
        ]

    def _load_rgb_frame(self, key, file_client=None):
        """Load only RGB LQ/GT frames, leaving spike IO to the crop-aware path."""
        fc = file_client if file_client is not None else self._ensure_file_client()
        clip_name, frame_name = key.split('/')
        neighbor = int(frame_name)
        if self.is_lmdb:
            img_lq_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
        else:
            img_lq_path = self.lq_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
            img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

        img_bytes = fc.get(img_lq_path, 'lq')
        img_lq = utils_video.imfrombytes(img_bytes, float32=True)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        img_bytes = fc.get(img_gt_path, 'gt')
        img_gt = utils_video.imfrombytes(img_bytes, float32=True)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'gt_path': img_gt_path,
        }

    def _load_raw_frame(self, key, file_client=None):
        """Load a single raw RGB+Spike frame triplet from disk/LMDB without augmentation."""
        clip_name, frame_name = key.split('/')
        neighbor = int(frame_name)
        sample = self._load_rgb_frame(key, file_client=file_client)

        # get Spike data
        spike_file = self.spike_root / clip_name / 'spike' / f'{neighbor:{self.filename_tmpl}}.dat'
        if spike_file.exists():
            spike_voxel = self._load_spike_voxel(clip_name, neighbor, spike_file)
        else:
            # If spike file doesn't exist, create zeros
            spike_voxel = np.zeros((self.spike_channels, self.spike_h, self.spike_w), dtype=np.float32)

        sample['spike'] = spike_voxel
        return sample

    def _load_spike_voxel(self, clip_name, neighbor, spike_file, crop=None):
        if self.use_precomputed_spike:
            precomputed = self._load_precomputed_spike_voxel(clip_name, neighbor, crop=crop)
            if precomputed is not None:
                return precomputed

        if crop is not None and not spike_file.exists():
            return np.zeros((self.spike_channels, crop['height'], crop['width']), dtype=np.float32)

        spike_stream = SpikeStream(
            offline=True,
            filepath=str(spike_file),
            spike_h=self.spike_h,
            spike_w=self.spike_w,
            print_dat_detail=False,
        )
        spike_matrix = spike_stream.get_spike_matrix(flipud=self.spike_flipud)  # (T, H, W)
        if self.spike_reconstruction in {'middle_tfp', 'middle-tfp'}:
            spike_frame = self._middle_tfp_reconstructor(spike_matrix)  # (H, W)
            return spike_frame[np.newaxis, ...].astype(np.float32)  # (1, H, W)
        if self.spike_reconstruction == 'snn':
            spike_frame = self._snn_reconstructor(spike_matrix)  # (H, W)
            return spike_frame[np.newaxis, ...].astype(np.float32)  # (1, H, W)
        if self.spike_representation == 'raw_window':
            raw_window = extract_centered_raw_window(
                spike_matrix,
                window_length=self.raw_window_length,
            )
            if crop is not None:
                raw_window = self._apply_chw_crop(raw_window, crop)
            return raw_window
        return voxelize_spikes_tfp(
            spike_matrix,
            num_channels=self.spike_channels,
            device=self._select_tfp_device(),
            half_win_length=self.tfp_half_win_length,
        )  # (S, H, W)

    def _build_precomputed_spike_base_path(self, clip_name, frame_idx):
        if self.spike_representation == 'raw_window':
            dir_name = f'raw_window_l{self.raw_window_length}'
        else:
            dir_name = f'tfp_b{self.spike_channels}_hw{self.tfp_half_win_length}'
        frame_name = f'{frame_idx:{self.filename_tmpl}}'
        return self.precomputed_spike_root / clip_name / dir_name / frame_name

    def _load_precomputed_spike_voxel(self, clip_name, frame_idx, crop=None):
        base_path = self._build_precomputed_spike_base_path(clip_name, frame_idx)
        if self.precomputed_spike_format == 'npy':
            path = base_path.with_suffix('.npy')
            if not path.exists():
                warn_key = (clip_name, 'npy')
                if warn_key not in self._precomputed_spike_warned:
                    self._precomputed_spike_warned.add(warn_key)
                    print(
                        f"[TRAIN_DATASET] precomputed spike miss for clip={clip_name}: "
                        f"expected {path}; falling back to raw .dat reconstruction.",
                        flush=True,
                    )
                return None
            if crop is not None and self.spike_representation == 'raw_window':
                spike_voxel = self._apply_chw_crop(np.load(path, mmap_mode='r'), crop)
            else:
                spike_voxel = np.load(path)
        elif self.precomputed_spike_format == 'npz':
            path = base_path.with_suffix('.npz')
            if not path.exists():
                warn_key = (clip_name, 'npz')
                if warn_key not in self._precomputed_spike_warned:
                    self._precomputed_spike_warned.add(warn_key)
                    print(
                        f"[TRAIN_DATASET] precomputed spike miss for clip={clip_name}: "
                        f"expected {path}; falling back to raw .dat reconstruction.",
                        flush=True,
                    )
                return None
            with np.load(path) as data:
                spike_voxel = data['spike_voxel']
        else:
            raise ValueError(f"Unsupported precomputed spike format: {self.precomputed_spike_format!r}")

        spike_voxel_arr = np.asarray(spike_voxel)
        if self.spike_representation == 'raw_window':
            spike_voxel = spike_voxel_arr.astype(np.float32)
        elif np.issubdtype(spike_voxel_arr.dtype, np.integer):
            # Precomputed uint8 artifacts store the original 0..255 TFP output losslessly.
            spike_voxel = spike_voxel_arr.astype(np.float32) / 255.0
        else:
            spike_voxel = spike_voxel_arr.astype(np.float32)
        if spike_voxel.ndim != 3:
            raise ValueError(
                f"Precomputed spike voxel must be [C,H,W], got {spike_voxel.shape} from {path}"
            )
        if spike_voxel.shape[0] != self.spike_channels:
            raise ValueError(
                f"Precomputed spike voxel channels mismatch: expected {self.spike_channels}, "
                f"got {spike_voxel.shape[0]} from {path}"
            )
        return spike_voxel

    def _validate_raw_rgb_spike_pair(self, rgb_hwc, spike_hwc):
        if rgb_hwc.ndim != 3 or spike_hwc.ndim != 3:
            raise ValueError(
                f"[TrainDatasetRGBSpike] Raw rgb/spike tensors must be HWC. "
                f"Got rgb ndim={rgb_hwc.ndim}, spike ndim={spike_hwc.ndim}."
            )
        if rgb_hwc.shape[:2] != spike_hwc.shape[:2]:
            raise ValueError(
                f"[TrainDatasetRGBSpike] dual pack mode requires matching spatial shape before concat: "
                f"rgb={rgb_hwc.shape[:2]}, spike={spike_hwc.shape[:2]}."
            )
        if spike_hwc.shape[2] != self.spike_channels:
            raise ValueError(
                f"[TrainDatasetRGBSpike] dual spike channels mismatch before concat: "
                f"expected {self.spike_channels}, got {spike_hwc.shape[2]}."
            )


    def _flow_module_name_from_opt(self, opt):
        netg = opt.get('netG', {}) if isinstance(opt, dict) else {}
        if isinstance(netg, dict):
            optical_flow = netg.get('optical_flow', {})
            if isinstance(optical_flow, dict):
                return str(optical_flow.get('module', '')).strip().lower()
        return str(opt.get('optical_flow_module', '')).strip().lower() if isinstance(opt, dict) else ''

    def _parse_spike_flow_config(self, opt, optical_flow_module=''):
        spike_flow_cfg = opt.get('spike_flow', {}) if isinstance(opt.get('spike_flow', {}), dict) else {}
        self.spike_flow_representation = str(spike_flow_cfg.get('representation', '')).strip().lower()
        self.spike_flow_dt = int(spike_flow_cfg.get('dt', 10))
        self.spike_flow_root = spike_flow_cfg.get('root', 'auto')
        self.spike_flow_subframes = int(spike_flow_cfg.get('subframes', 1))
        self.spike_flow_format = str(spike_flow_cfg.get('format', 'auto')).strip().lower()
        self.use_encoding25_flow = self.spike_flow_representation == 'encoding25'

        flow_module = str(optical_flow_module or '').strip().lower()
        if flow_module == 'spike_flow':
            flow_module = 'scflow'
        if flow_module == 'scflow' and self.spike_flow_representation != 'encoding25':
            raise ValueError(
                "SCFlow strict mode requires spike_flow.representation='encoding25'. "
                f"Got {self.spike_flow_representation!r}."
            )

    def _load_encoded_flow_spike(self, clip_name, frame_idx):
        flow_root = self.spike_root if str(self.spike_flow_root).strip().lower() == 'auto' else Path(self.spike_flow_root)
        frame_name = f'{frame_idx:{self.filename_tmpl}}'
        spike_flow_format = getattr(self, 'spike_flow_format', 'auto')
        spike_h = getattr(self, 'spike_h', 360)
        spike_w = getattr(self, 'spike_w', 640)

        if self.spike_flow_subframes > 1:
            dir_name = f'encoding25_dt{self.spike_flow_dt}_s{self.spike_flow_subframes}'
            base_path = flow_root / clip_name / dir_name / frame_name
            try:
                arr = load_encoding25_artifact_with_shape(
                    base_path,
                    artifact_format=spike_flow_format,
                    num_subframes=self.spike_flow_subframes,
                    spike_h=spike_h,
                    spike_w=spike_w,
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Missing subframe encoding25 artifact: {base_path}.npy or .dat. "
                    "Run prepare_scflow_encoding25.py --subframes first."
                ) from exc
            validate_subframes_tensor(arr, self.spike_flow_subframes)
            return arr
        else:
            base_path = flow_root / clip_name / f'encoding25_dt{self.spike_flow_dt}' / frame_name
            try:
                arr = load_encoding25_artifact_with_shape(
                    base_path,
                    artifact_format=spike_flow_format,
                    num_subframes=1,
                    spike_h=spike_h,
                    spike_w=spike_w,
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Missing encoding25 artifact: {base_path}.npy or .dat. "
                    "Run scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py first."
                ) from exc
            validate_encoding25_tensor(arr)
            return arr

    def _validate_dual_tensor_contract(self, l_rgb, l_spike):
        if l_rgb.ndim != 4 or l_spike.ndim != 4:
            raise ValueError(
                f"[TrainDatasetRGBSpike] dual tensors must be [T,C,H,W], got "
                f"L_rgb ndim={l_rgb.ndim}, L_spike ndim={l_spike.ndim}."
            )
        if l_rgb.size(1) != 3:
            raise ValueError(
                f"[TrainDatasetRGBSpike] dual RGB tensor should have 3 channels, got {l_rgb.size(1)}."
            )
        if l_spike.size(1) != self.spike_channels:
            raise ValueError(
                f"[TrainDatasetRGBSpike] dual spike channels mismatch: "
                f"expected {self.spike_channels}, got {l_spike.size(1)}."
            )
        if l_rgb.shape[0] != l_spike.shape[0] or l_rgb.shape[2:] != l_spike.shape[2:]:
            raise ValueError(
                f"[TrainDatasetRGBSpike] dual temporal/spatial mismatch between L_rgb {l_rgb.shape} "
                f"and L_spike {l_spike.shape}."
            )

    def _normalize_single_tfp_device(self, device_value):
        """Normalize a single TFP device specification."""
        if device_value is None:
            return 'cpu'
        device_str = str(device_value).strip()
        if not device_str:
            return 'cpu'
        return self._sanitize_device_string(device_str)

    def _normalize_tfp_device_pool(self, pool_value):
        """Normalize a list of TFP devices for multi-GPU reconstruction."""
        if not pool_value:
            return []
        if isinstance(pool_value, str):
            pool_value = [pool_value]
        normalized = []
        for dev in pool_value:
            sanitized = self._sanitize_device_string(str(dev).strip())
            if sanitized:
                normalized.append(sanitized)
        # Remove duplicates while preserving order to avoid oversubscribing the same GPU.
        seen = set()
        unique_devices = []
        for dev in normalized:
            if dev not in seen:
                unique_devices.append(dev)
                seen.add(dev)
        return unique_devices

    def _sanitize_device_string(self, device_str):
        """Validate device string and gracefully fall back to a safe option."""
        if device_str.lower() in {'auto', 'auto_cuda', 'cuda:auto'}:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
                visible = torch.cuda.device_count()
                return f'cuda:{local_rank % visible}'
            print('TFP device "auto" requested but CUDA is unavailable. Falling back to CPU.')
            return 'cpu'

        if device_str.lower().startswith('cuda'):
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                print(f'TFP device "{device_str}" requested but CUDA is unavailable. Falling back to CPU.')
                return 'cpu'
            parts = device_str.split(':')
            if len(parts) == 1 or parts[1] == '':
                return 'cuda:0'
            try:
                dev_idx = int(parts[1])
            except ValueError:
                print(f'Invalid CUDA device spec "{device_str}". Falling back to cuda:0.')
                dev_idx = 0
            visible = torch.cuda.device_count()
            if dev_idx >= visible:
                print(f'CUDA device index {dev_idx} is out of range (visible: {visible}). '
                      f'Using cuda:{dev_idx % max(1, visible)} instead.')
                dev_idx = dev_idx % max(1, visible)
            return f'cuda:{dev_idx}'

        return 'cpu'

    def _select_tfp_device(self):
        """Choose a device for the current worker/process when multi-device mode is enabled."""
        if not self._tfp_device_pool:
            return self.tfp_device or 'cpu'

        worker_info = get_worker_info()
        if worker_info is not None:
            idx = worker_info.id % len(self._tfp_device_pool)
            return self._tfp_device_pool[idx]

        # Fall back to LOCAL_RANK or RANK when not inside a DataLoader worker.
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank is not None:
            idx = int(local_rank) % len(self._tfp_device_pool)
            return self._tfp_device_pool[idx]

        rank = os.environ.get('RANK')
        if rank is not None:
            idx = int(rank) % len(self._tfp_device_pool)
            return self._tfp_device_pool[idx]

        # Default to the first configured device.
        return self._tfp_device_pool[0]

    def _build_norm_stats(self, cfg_value, num_channels, preset=None):
        """Return per-channel mean/std tensors when optional normalization is requested."""
        if cfg_value in (None, False) or num_channels == 0:
            return None

        if isinstance(cfg_value, str):
            key = cfg_value.lower()
            if key == 'imagenet' and preset == 'imagenet' and num_channels == 3:
                mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
                std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
            else:
                raise ValueError(f'Unsupported normalization preset: {cfg_value}')
            mean = mean.view(1, num_channels, 1, 1)
            std = std.view(1, num_channels, 1, 1)
        elif isinstance(cfg_value, dict):
            mean = self._to_channel_tensor(cfg_value.get('mean', 0.0), num_channels, 'mean')
            std = self._to_channel_tensor(cfg_value.get('std', 1.0), num_channels, 'std')
        else:
            raise ValueError(f'Unsupported normalization config type: {type(cfg_value)}')

        return {'mean': mean, 'std': std}

    def _to_channel_tensor(self, value, num_channels, label):
        tensor = torch.tensor(value, dtype=torch.float32).flatten()
        if tensor.numel() == 1:
            tensor = tensor.repeat(num_channels)
        elif num_channels % tensor.numel() == 0:
            # Allow repeating a shorter pattern (e.g., 4 values for 8 channels).
            repeat_times = num_channels // tensor.numel()
            tensor = tensor.repeat(repeat_times)
        if tensor.numel() != num_channels:
            raise ValueError(f'Normalization {label} expected {num_channels} values, got {tensor.numel()}')
        return tensor.view(1, num_channels, 1, 1)

    def _apply_channel_normalization(self, tensor):
        """Apply RGB ImageNet-style normalization and optional spike scaling."""
        if self.rgb_norm_stats is not None:
            tensor[:, :3, :, :] = (tensor[:, :3, :, :] - self.rgb_norm_stats['mean']) / self.rgb_norm_stats['std']
        if self.spike_norm_stats is not None and tensor.size(1) > 3:
            tensor[:, 3:, :, :] = (tensor[:, 3:, :, :] - self.spike_norm_stats['mean']) / self.spike_norm_stats['std']
        return tensor
