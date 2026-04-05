import numpy as np
import os
import random
import torch
from pathlib import Path
import torch.utils.data as data
from torch.utils.data import get_worker_info
import cv2

import utils.utils_video as utils_video
from data.spike_recc import SpikeStream, voxelize_spikes_tfp
from data.spike_recc.middle_tfp.reconstructor import MiddleTFPReconstructor
from data.spike_recc.snn.reconstructor import SNNReconstructor


class TrainDatasetRGBSpike(data.Dataset):
    """Video dataset for training recurrent networks with RGB + Spike data.

    This dataset extends TrainDataset to support loading both
    RGB frames and corresponding Spike camera data, then concatenates them
    as input channels.

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
            spike_channels (int): Number of spike channels after voxelization. Default: 1.
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
        self.filename_ext = opt.get('filename_ext', 'png')
        self.num_frame = opt['num_frame']

        # Spike configuration
        self.spike_h = opt.get('spike_h', 360)
        self.spike_w = opt.get('spike_w', 640)
        self.spike_channels = opt.get('spike_channels', 4) # Default updated to 4 for TFP
        self.spike_flipud = opt.get('spike_flipud', True)
        self.tfp_half_win_length = opt.get('tfp_half_win_length', 20)
        spike_reconstruction_cfg = opt.get('spike_reconstruction', 'spikecv_tfp')
        if isinstance(spike_reconstruction_cfg, dict):
            self.spike_reconstruction = str(spike_reconstruction_cfg.get('type', 'spikecv_tfp')).strip().lower()
            self.middle_tfp_center = int(spike_reconstruction_cfg.get('middle_tfp_center', 44))
        else:
            self.spike_reconstruction = str(spike_reconstruction_cfg).strip().lower()
            self.middle_tfp_center = int(opt.get('middle_tfp_center', 44))

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
            if self.spike_channels != 1:
                self.spike_channels = 1
            self._middle_tfp_reconstructor = MiddleTFPReconstructor(
                spike_h=self.spike_h,
                spike_w=self.spike_w,
                center=self.middle_tfp_center,
            )
        elif self.spike_reconstruction == 'snn':
            if self.spike_channels != 1:
                self.spike_channels = 1
            snn_cfg = spike_reconstruction_cfg if isinstance(spike_reconstruction_cfg, dict) else {}
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
        
        img_gt_path_reference = None
        for neighbor in neighbor_list:
            neighbor_key = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            sample = self._load_raw_frame(neighbor_key)
            img_lqs.append(sample['lq'])
            img_gts.append(sample['gt'])
            spike_voxels.append(sample['spike'])
            img_gt_path_reference = sample['gt_path']

        # randomly crop RGB frames
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path_reference)

        # Resize spike voxels to match the cropped RGB size
        cropped_h, cropped_w = img_lqs[0].shape[:2]
        spike_voxels_resized = []
        for spike_voxel in spike_voxels:
            # spike_voxel: (S, H, W)
            spike_voxel_resized = []
            for ch in range(self.spike_channels):
                spike_ch = spike_voxel[ch]  # (H, W)
                spike_ch_resized = cv2.resize(spike_ch, (cropped_w, cropped_h), interpolation=cv2.INTER_LINEAR)
                spike_voxel_resized.append(spike_ch_resized)
            spike_voxel_resized = np.stack(spike_voxel_resized, axis=0)  # (S, H, W)
            spike_voxels_resized.append(spike_voxel_resized)

        # Concatenate RGB and Spike channels
        # Channel Order: 
        #   0~2: RGB (0-1 float)
        #   3~6: Spike TFP Voxels (0-1 float, 4 channels)
        # Total: 7 channels
        img_lqs_with_spike = []
        for img_lq, spike_voxel in zip(img_lqs, spike_voxels_resized):
            # img_lq: (H, W, 3), spike_voxel: (S, H, W)
            # Convert spike_voxel to (H, W, S)
            spike_voxel_hwc = np.transpose(spike_voxel, (1, 2, 0))  # (H, W, S)
            # Concatenate along channel dimension
            img_lq_with_spike = np.concatenate([img_lq, spike_voxel_hwc], axis=2)  # (H, W, 3+S)
            img_lqs_with_spike.append(img_lq_with_spike)

        # augmentation - flip, rotate
        img_lqs_with_spike.extend(img_gts)
        img_results = utils_video.augment(img_lqs_with_spike, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results, bgr2rgb=False)
        img_gts = torch.stack(img_results[len(img_lqs_with_spike) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs_with_spike) // 2], dim=0)
        if img_lqs.size(1) != self.expected_lq_channels:
            raise ValueError(
                f"[TrainDatasetRGBSpike] Expected {self.expected_lq_channels} channels "
                f"but received {img_lqs.size(1)} (tensor shape: {img_lqs.shape}). "
                "Double-check spike voxel stacking and augmentation."
            )
        img_lqs = self._apply_channel_normalization(img_lqs)

        # img_lqs: (t, c, h, w) where c = 3 + spike_channels
        # img_gts: (t, c, h, w) where c = 3
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)

    def _ensure_file_client(self):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_type, **self.io_backend_opt)
        return self.file_client

    def _new_file_client(self):
        """Create an isolated FileClient instance (for threaded cache warmup)."""
        return utils_video.FileClient(self.io_backend_type, **self.io_backend_opt)


    def _load_raw_frame(self, key, file_client=None):
        """Load a single raw RGB+Spike frame triplet from disk/LMDB without augmentation."""
        fc = file_client if file_client is not None else self._ensure_file_client()
        clip_name, frame_name = key.split('/')
        neighbor = int(frame_name)
        if self.is_lmdb:
            img_lq_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
        else:
            img_lq_path = self.lq_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
            img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

        # get LQ
        img_bytes = fc.get(img_lq_path, 'lq')
        img_lq = utils_video.imfrombytes(img_bytes, float32=True)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # get GT
        img_bytes = fc.get(img_gt_path, 'gt')
        img_gt = utils_video.imfrombytes(img_bytes, float32=True)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # get Spike data
        spike_file = self.spike_root / clip_name / 'spike' / f'{neighbor:{self.filename_tmpl}}.dat'
        if spike_file.exists():
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
                spike_voxel = spike_frame[np.newaxis, ...].astype(np.float32)  # (1, H, W)
            elif self.spike_reconstruction == 'snn':
                spike_frame = self._snn_reconstructor(spike_matrix)  # (H, W)
                spike_voxel = spike_frame[np.newaxis, ...].astype(np.float32)  # (1, H, W)
            else:
                spike_voxel = voxelize_spikes_tfp(
                    spike_matrix,
                    num_channels=self.spike_channels,
                    device=self._select_tfp_device(),
                    half_win_length=self.tfp_half_win_length,
                )  # (S, H, W)
        else:
            # If spike file doesn't exist, create zeros
            spike_voxel = np.zeros((self.spike_channels, self.spike_h, self.spike_w), dtype=np.float32)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'spike': spike_voxel,
            'gt_path': img_gt_path,
        }

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

