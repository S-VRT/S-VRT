import numpy as np
import random
import torch
from pathlib import Path
import torch.utils.data as data
import cv2

import utils.utils_video as utils_video
from utils.spike_loader import SpikeStreamSimple, voxelize_spikes


class VideoRecurrentTrainDatasetRGBSpike(data.Dataset):
    """Video dataset for training recurrent networks with RGB + Spike data.

    This dataset extends VideoRecurrentTrainDataset to support loading both
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
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainDatasetRGBSpike, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.spike_root = Path(opt['dataroot_spike'])
        self.filename_tmpl = opt.get('filename_tmpl', '08d')
        self.filename_ext = opt.get('filename_ext', 'png')
        self.num_frame = opt['num_frame']

        # Spike configuration
        self.spike_h = opt.get('spike_h', 250)
        self.spike_w = opt.get('spike_w', 400)
        self.spike_channels = opt.get('spike_channels', 1)
        self.spike_flipud = opt.get('spike_flipud', True)

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
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

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
        
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
                img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

            # get Spike data
            spike_file = self.spike_root / clip_name / 'spike' / f'{neighbor:{self.filename_tmpl}}.dat'
            if spike_file.exists():
                spike_stream = SpikeStreamSimple(
                    str(spike_file),
                    spike_h=self.spike_h,
                    spike_w=self.spike_w,
                    print_dat_detail=False
                )
                spike_matrix = spike_stream.get_spike_matrix(flipud=self.spike_flipud)  # (T, H, W)
                spike_voxel = voxelize_spikes(spike_matrix, num_channels=self.spike_channels)  # (S, H, W)
                spike_voxels.append(spike_voxel)
            else:
                # If spike file doesn't exist, create zeros
                spike_voxel = np.zeros((self.spike_channels, self.spike_h, self.spike_w), dtype=np.float32)
                spike_voxels.append(spike_voxel)

        # randomly crop RGB frames
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)

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

        img_results = utils_video.img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs_with_spike) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs_with_spike) // 2], dim=0)

        # img_lqs: (t, c, h, w) where c = 3 + spike_channels
        # img_gts: (t, c, h, w) where c = 3
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)

