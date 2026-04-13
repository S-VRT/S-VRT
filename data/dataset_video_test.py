import glob
import os
import numpy as np
import torch
from os import path as osp
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import cv2

import utils.utils_video as utils_video
from data.spike_recc import SpikeStream, voxelize_spikes_tfp
from data.spike_recc.middle_tfp.reconstructor import MiddleTFPReconstructor
from data.spike_recc.snn.reconstructor import SNNReconstructor
from data.spike_recc.encoding25 import validate_encoding25_tensor


class TestDataset(data.Dataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames. Modified from
    https://github.com/xinntao/BasicSR/blob/master/basicsr/data/reds_dataset.py

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(TestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = utils_video.read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))
        self.sigma = opt['sigma'] / 255. if 'sigma' in opt and opt['sigma'] is not None else 0 # for non-blind video denoising

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.sigma:
        # for non-blind video denoising
            if self.cache_data:
                imgs_gt = self.imgs_gt[folder]
            else:
                imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

            torch.manual_seed(0)
            noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
            noise = torch.normal(mean=0, std=noise_level.expand_as(imgs_gt))
            imgs_lq = imgs_gt + noise
            t, _, h, w = imgs_lq.shape
            imgs_lq = torch.cat([imgs_lq, noise_level.expand(t, 1, h, w)], 1)
        else:
        # for video sr and deblurring
            if self.cache_data:
                imgs_lq = self.imgs_lq[folder]
                imgs_gt = self.imgs_gt[folder]
            else:
                imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])
                imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

        return {
            'L': imgs_lq,
            'H': imgs_gt,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)


class TrainDatasetRGBSpike(data.Dataset):
    """Video test dataset that concatenates RGB frames with Spike data channels."""

    def __init__(self, opt):
        super(TrainDatasetRGBSpike, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        # Spike-specific configuration
        self.spike_root = opt['dataroot_spike']
        self._parse_spike_flow_config(opt, optical_flow_module=self._flow_module_name_from_opt(opt))
        self.spike_h = opt.get('spike_h', 250)
        self.spike_w = opt.get('spike_w', 400)
        self.spike_channels = opt.get('spike_channels', 4)  # Default updated to 4 for TFP
        self.spike_flipud = opt.get('spike_flipud', True)
        self.spike_folder = opt.get('spike_folder_name', 'spike')
        self.spike_ext = opt.get('spike_filename_ext', 'dat')
        self.tfp_half_win_length = opt.get('tfp_half_win_length', 20)
        spike_reconstruction_cfg = opt.get('spike_reconstruction', 'spikecv_tfp')
        if isinstance(spike_reconstruction_cfg, dict):
            self.spike_reconstruction = str(spike_reconstruction_cfg.get('type', 'spikecv_tfp')).strip().lower()
            self.middle_tfp_center = int(spike_reconstruction_cfg.get('middle_tfp_center', 44))
        else:
            self.spike_reconstruction = str(spike_reconstruction_cfg).strip().lower()
            self.middle_tfp_center = int(opt.get('middle_tfp_center', 44))
        
        requested_tfp_device = str(opt.get('tfp_device', 'cpu'))
        if requested_tfp_device.startswith('cuda') and not torch.cuda.is_available():
            print("Requested CUDA device for TFP but CUDA is unavailable. Falling back to CPU.")
            requested_tfp_device = 'cpu'
        self.tfp_device = requested_tfp_device
        
        # Initialize spike reconstructors
        self._middle_tfp_reconstructor = None
        self._snn_reconstructor = None
        
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

        self.imgs_lq, self.imgs_gt = {}, {}
        self.spike_paths = {}
        self.spike_cache = {}
        self.flow_spike_cache = {}
        self.frame_basenames = {}

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            self.frame_basenames[subfolder_name] = [osp.splitext(osp.basename(p))[0] for p in img_paths_lq]

            spike_folder = osp.join(self.spike_root, subfolder_name, self.spike_folder)
            spike_paths = []
            for img_path in img_paths_lq:
                base = osp.splitext(osp.basename(img_path))[0]
                spike_paths.append(osp.join(spike_folder, f'{base}.{self.spike_ext}'))
            self.spike_paths[subfolder_name] = spike_paths

            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTrainDatasetRGBSpike...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = utils_video.read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

        self.folders = sorted(list(set(self.data_info['folder'])))
        
        # 诊断：打印数据集初始化时找到的文件夹
        print(f'\n{"="*80}')
        print(f'[Dataset Initialization - TrainDatasetRGBSpike]')
        print(f'  Found {len(self.folders)} folders: {self.folders}')
        print(f'  Total samples (frames): {len(self.data_info["folder"])}')
        print(f'  LQ root: {self.lq_root}')
        print(f'  GT root: {self.gt_root}')
        print(f'  Spike root: {self.spike_root}')
        print(f'{"="*80}\n')

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])
            imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

        t, _, h, w = imgs_lq.shape
        spike_seq = self._get_spike_sequence(folder, h, w, t)
        imgs_lq = torch.cat([imgs_lq, spike_seq], dim=1)
        flow_seq = None
        if self.use_encoding25_flow:
            flow_seq = self._get_flow_spike_sequence(folder, h, w, t)
        if imgs_lq.size(1) != self.expected_lq_channels:
            raise ValueError(
                f"[TrainDatasetRGBSpike] Expected {self.expected_lq_channels} channels "
                f"but found {imgs_lq.size(1)} for folder {folder}."
            )
        imgs_lq = self._apply_channel_normalization(imgs_lq)

        sample = {
            'L': imgs_lq,
            'H': imgs_gt,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }
        if flow_seq is not None:
            if flow_seq.ndim != 4 or flow_seq.size(1) != 25:
                raise ValueError(f"Expected L_flow_spike [T,25,H,W], got {tuple(flow_seq.shape)}")
            sample['L_flow_spike'] = flow_seq
        return sample

    def _get_spike_sequence(self, folder, target_h, target_w, expected_len):
        if folder in self.spike_cache:
            return self.spike_cache[folder]

        spike_tensors = []
        spike_paths = self.spike_paths.get(folder, [])
        if spike_paths and len(spike_paths) != expected_len:
            raise ValueError(f'Spike sequence length mismatch for {folder}: '
                             f'{len(spike_paths)} vs {expected_len}')

        for spike_path in spike_paths:
            spike_voxel = self._load_spike_voxel(spike_path)
            spike_voxel = self._resize_spike_voxel(spike_voxel, target_h, target_w)
            spike_tensors.append(torch.from_numpy(spike_voxel).float())

        spike_seq = torch.stack(spike_tensors, dim=0)
        self.spike_cache[folder] = spike_seq
        return spike_seq


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
        self.use_encoding25_flow = self.spike_flow_representation == 'encoding25'

        flow_module = str(optical_flow_module or '').strip().lower()
        if flow_module == 'spike_flow':
            flow_module = 'scflow'
        if flow_module == 'scflow' and self.spike_flow_representation != 'encoding25':
            raise ValueError(
                "SCFlow strict mode requires spike_flow.representation='encoding25'. "
                f"Got {self.spike_flow_representation!r}."
            )

    def _get_flow_spike_sequence(self, folder, target_h, target_w, expected_len):
        if folder in self.flow_spike_cache:
            return self.flow_spike_cache[folder]

        flow_root = self.spike_root if str(self.spike_flow_root).strip().lower() == 'auto' else self.spike_flow_root
        flow_clip_dir = os.path.join(flow_root, folder, f'encoding25_dt{self.spike_flow_dt}')

        frame_names = self.frame_basenames.get(folder, [])
        if frame_names and len(frame_names) != expected_len:
            raise ValueError(f'Frame sequence length mismatch for {folder}: {len(frame_names)} vs {expected_len}')

        flow_tensors = []
        for frame_name in frame_names:
            flow_path = os.path.join(flow_clip_dir, f'{frame_name}.npy')
            if not os.path.exists(flow_path):
                raise ValueError(
                    f"Missing encoding25 artifact: {flow_path}. "
                    "Run scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py first."
                )
            arr = np.load(flow_path).astype(np.float32)
            validate_encoding25_tensor(arr)
            resized = []
            for ch in range(arr.shape[0]):
                resized.append(cv2.resize(arr[ch], (target_w, target_h), interpolation=cv2.INTER_LINEAR))
            flow_tensors.append(torch.from_numpy(np.stack(resized, axis=0).astype(np.float32)))

        if not flow_tensors:
            raise ValueError(f'No frame names available to build L_flow_spike for folder {folder}.')
        flow_seq = torch.stack(flow_tensors, dim=0)
        self.flow_spike_cache[folder] = flow_seq
        return flow_seq

    def _load_spike_voxel(self, spike_path):
        if os.path.exists(spike_path):
            try:
                spike_stream = SpikeStream(
                    offline=True,
                    filepath=spike_path,
                    spike_h=self.spike_h,
                    spike_w=self.spike_w,
                    print_dat_detail=False
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
                        device=self.tfp_device,
                        half_win_length=self.tfp_half_win_length,
                    )  # (S, H, W)
            except Exception as err:
                print(f'Failed to load spike data {spike_path}: {err}. Using zeros.')
                # Use correct channel count based on reconstruction method
                spike_voxel = np.zeros((self.spike_channels, self.spike_h, self.spike_w), dtype=np.float32)
        else:
            # Use correct channel count based on reconstruction method
            spike_voxel = np.zeros((self.spike_channels, self.spike_h, self.spike_w), dtype=np.float32)
        return spike_voxel.astype(np.float32)

    def _resize_spike_voxel(self, spike_voxel, target_h, target_w):
        resized_channels = []
        for ch in range(self.spike_channels):
            resized_channels.append(
                cv2.resize(spike_voxel[ch], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            )
        return np.stack(resized_channels, axis=0)

    def _build_norm_stats(self, cfg_value, num_channels, preset=None):
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
        if self.rgb_norm_stats is not None:
            tensor[:, :3, :, :] = (tensor[:, :3, :, :] - self.rgb_norm_stats['mean']) / self.rgb_norm_stats['std']
        if self.spike_norm_stats is not None and tensor.size(1) > 3:
            tensor[:, 3:, :, :] = (tensor[:, 3:, :, :] - self.spike_norm_stats['mean']) / self.spike_norm_stats['std']
        return tensor


class SingleTestDataset(data.Dataset):
    """Single video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames (only input LQ path).

    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(SingleTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.lq_root = opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq = {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))

        for subfolder_lq in subfolders_lq:
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))

            max_idx = len(img_paths_lq)

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
        else:
            imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])

        return {
            'L': imgs_lq,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)
