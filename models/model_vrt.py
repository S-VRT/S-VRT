from collections import OrderedDict
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_plain import ModelPlain
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss
from data.spike_recc.encoding25 import load_encoding25_artifact_with_shape

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


def _tensor_gb(tensor):
    if tensor is None:
        return 0.0
    return tensor.numel() * tensor.element_size() / (1024 ** 3)


class ModelVRT(ModelPlain):
    """Train video restoration  with pixel loss"""
    def __init__(self, opt):
        super(ModelVRT, self).__init__(opt)
        self.fix_iter = self.opt_train.get('fix_iter', 0)
        self.fix_keys = self.opt_train.get('fix_keys', [])
        self.fix_unflagged = True

    def feed_data(self, data, need_H=True):
        self.L_flow_spike_meta = None
        if self._flow_module_name() == 'scflow' and 'L_flow_spike_meta' in data and 'L_flow_spike' not in data:
            with self.timer.timer('data_load'):
                self.L = self._build_model_input_tensor(data).to(self.device)
                self._assert_lq_channels(self.L, 'Feed Data')
                self.L_flow_spike = None
                self.L_flow_spike_meta = self._normalize_flow_spike_meta(data['L_flow_spike_meta'])
                if need_H:
                    self.H = data['H'].to(self.device)
            return

        # Keep VRT data ingress aligned with ModelPlain dual/concat input routing.
        super(ModelVRT, self).feed_data(data, need_H=need_H)

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        self.fix_keys = self.opt_train.get('fix_keys', [])
        fix_lr_mul = self.opt_train.get('fix_lr_mul', 1.0)
        use_split_lr = len(self.fix_keys) > 0 and fix_lr_mul != 1.0
        if use_split_lr:
            print(f'Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.')
            normal_params = []
            flow_params = []
            for name, param in self.netG.named_parameters():
                if not param.requires_grad:
                    continue
                if any([key in name for key in self.fix_keys]):
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            G_optim_params = [
                {'params': normal_params, 'lr': self.opt_train['G_optimizer_lr']},
                {'params': flow_params, 'lr': self.opt_train['G_optimizer_lr'] * fix_lr_mul},
            ]
            if self.opt_train['G_optimizer_type'] == 'adam':
                self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                        betas=self.opt_train['G_optimizer_betas'],
                                        weight_decay=self.opt_train['G_optimizer_wd'])
            else:
                raise NotImplementedError
        else:
            super(ModelVRT, self).define_optimizer()

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        if self.fix_iter:
            if self.fix_unflagged and current_step < self.fix_iter:
                print(f'Fix keys: {self.fix_keys} for the first {self.fix_iter} iters.')
                self.fix_unflagged = False
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_step == self.fix_iter:
                print(f'Train all the parameters from {self.fix_iter} iters.')
                self.netG.requires_grad_(True)
                # Re-wrap DDP if using static graph to capture new graph structure
                if self.opt.get('dist', False) and self.opt.get('use_static_graph', False):
                    print('Re-wrapping DDP for static graph change...')
                    self.netG = self.model_to_device(self.get_bare_model(self.netG))

        super(ModelVRT, self).optimize_parameters(current_step)

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        n = self.L.size(1)
        self.netG.eval()

        pad_seq = self.opt_train.get('pad_seq', False)
        flip_seq = self.opt_train.get('flip_seq', False)
        self.center_frame_only = self.opt_train.get('center_frame_only', False)

        if pad_seq:
            n = n + 1
            self.L = torch.cat([self.L, self.L[:, -1:, :, :, :]], dim=1)

        if flip_seq:
            self.L = torch.cat([self.L, self.L.flip(1)], dim=1)

        flow_spike = getattr(self, 'L_flow_spike', None)
        flow_spike_meta = getattr(self, 'L_flow_spike_meta', None)
        if pad_seq and flow_spike is not None:
            flow_spike = torch.cat([flow_spike, flow_spike[:, -1:, :, :, :]], dim=1)
        if flip_seq and flow_spike is not None:
            flow_spike = torch.cat([flow_spike, flow_spike.flip(1)], dim=1)
        if (pad_seq or flip_seq) and flow_spike_meta is not None:
            raise NotImplementedError("Lazy validation flow metadata does not support pad_seq/flip_seq.")

        with torch.no_grad():
            self.E = self._test_video(self.L, flow_spike=flow_spike, flow_spike_meta=flow_spike_meta)

        if flip_seq:
            output_1 = self.E[:, :n, :, :, :]
            output_2 = self.E[:, n:, :, :, :].flip(1)
            self.E = 0.5 * (output_1 + output_2)

        if pad_seq:
            n = n - 1
            self.E = self.E[:, :n, :, :, :]

        if self.center_frame_only:
            self.E = self.E[:, n // 2, :, :, :]

        self.netG.train()

    def _test_video(self, lq, flow_spike=None, flow_spike_meta=None):
        '''test the video as a whole or as clips (divided temporally). '''

        self._assert_lq_channels(lq, 'Test Video Input')
        print(
            f'[VAL_MODEL] _test_video start lq={tuple(lq.shape)} '
            f'flow_spike={None if flow_spike is None else tuple(flow_spike.shape)} '
            f'lazy_flow={flow_spike_meta is not None}'
        )

        num_frame_testing = self.opt['val'].get('num_frame_testing', 0)

        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = self.opt['scale']
            num_frame_overlapping = self.opt['val'].get('num_frame_overlapping', 2)
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt['netG'].get('nonblind_denoising', False) else c
            c_out = self.opt['netG'].get('out_chans', c)
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = None
            W = None

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                flow_spike_clip = None if flow_spike is None else flow_spike[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = self._test_clip(
                    lq_clip,
                    flow_spike=flow_spike_clip,
                    flow_spike_meta=flow_spike_meta,
                    temporal_offset=d_idx,
                    full_h=h,
                    full_w=w,
                )
                if E is None:
                    c_out = out_clip.size(2)
                    E = torch.zeros(b, d, c_out, h*sf, w*sf)
                    W = torch.zeros(b, d, 1, 1, 1)
                    print(
                        '[VAL_MODEL] _test_video allocated '
                        f'E={tuple(E.shape)} ({_tensor_gb(E):.2f} GB) '
                        f'W={tuple(W.shape)} ({_tensor_gb(W):.6f} GB)'
                    )
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = self.opt['netG'].get('window_size', [6,8,8])
            d_old = lq.size(1)
            d_pad = (d_old// window_size[0]+1)*window_size[0] - d_old
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
            if flow_spike is not None:
                flow_spike = torch.cat([flow_spike, torch.flip(flow_spike[:, -d_pad:, ...], [1])], 1)
            output = self._test_clip(lq, flow_spike=flow_spike, flow_spike_meta=flow_spike_meta)
            output = output[:, :d_old, :, :, :]

        return output

    def _test_clip(self, lq, flow_spike=None, flow_spike_meta=None, temporal_offset=0, full_h=None, full_w=None):
        ''' test the clip as a whole or as patches. '''

        self._assert_lq_channels(lq, 'Test Clip Input')
        print(
            f'[VAL_MODEL] _test_clip start lq={tuple(lq.shape)} '
            f'flow_spike={None if flow_spike is None else tuple(flow_spike.shape)} '
            f'lazy_flow={flow_spike_meta is not None}'
        )

        sf = self.opt['scale']
        window_size = self.opt['netG'].get('window_size', [6,8,8])
        size_patch_testing = self.opt['val'].get('size_patch_testing', 0)
        assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = int(self.opt['val'].get('size_patch_overlapping', 20))
            if overlap_size < 0 or overlap_size >= size_patch_testing:
                raise ValueError(
                    f"size_patch_overlapping must be in [0, size_patch_testing), "
                    f"got overlap={overlap_size}, patch={size_patch_testing}"
                )
            not_overlap_border = True

            # Test the same spatial patches as KAIR/VRT, but optionally group
            # independent patches into one GPU forward to reduce Python overhead.
            b, d, c, h, w = lq.size()
            if b != 1:
                raise NotImplementedError("Validation patch batching expects dataloader_batch_size=1.")
            c = c - 1 if self.opt['netG'].get('nonblind_denoising', False) else c
            c_out = self.opt['netG'].get('out_chans', c)
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
            w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
            patch_coords = [(h_idx, w_idx) for h_idx in h_idx_list for w_idx in w_idx_list]
            patch_batch_size = max(1, int(self.opt['val'].get('patch_batch_size', 1)))
            cache_flow_patches_cpu = bool(self.opt['val'].get('cache_flow_patches_cpu', False))
            print(
                f'[VAL_MODEL] _test_clip patches={len(patch_coords)} '
                f'patch_batch_size={patch_batch_size} patch={size_patch_testing} overlap={overlap_size} '
                f'cache_flow_patches_cpu={cache_flow_patches_cpu}'
            )
            E = None
            W = None
            lazy_flow_clip = None
            flow_patch_cache = None
            if flow_spike_meta is not None:
                lazy_flow_clip = self._load_lazy_flow_clip(flow_spike_meta, temporal_offset, d)
                if cache_flow_patches_cpu:
                    flow_patch_cache = [
                        self._crop_resize_lazy_flow_patch_cpu(
                            lazy_flow_clip,
                            h_idx,
                            w_idx,
                            size_patch_testing,
                            size_patch_testing,
                            full_h or h,
                            full_w or w,
                        )
                        for h_idx, w_idx in patch_coords
                    ]
                    cached_gb = sum(_tensor_gb(patch) for patch in flow_patch_cache)
                    print(
                        '[VAL_MODEL] _test_clip cached flow patches on CPU '
                        f'count={len(flow_patch_cache)} total={cached_gb:.2f} GB'
                    )

            for coord_start in range(0, len(patch_coords), patch_batch_size):
                coord_batch = patch_coords[coord_start:coord_start + patch_batch_size]
                in_patches = []
                flow_patches = []
                for patch_offset, (h_idx, w_idx) in enumerate(coord_batch):
                    in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                    in_patches.append(in_patch)
                    flow_patch = None
                    if flow_spike is not None:
                        flow_patch = flow_spike[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                    elif flow_patch_cache is not None:
                        flow_patch = flow_patch_cache[coord_start + patch_offset]
                    elif flow_spike_meta is not None:
                        flow_patch = self._crop_resize_lazy_flow_patch(
                            lazy_flow_clip,
                            h_idx,
                            w_idx,
                            size_patch_testing,
                            size_patch_testing,
                            full_h or h,
                            full_w or w,
                        )
                    if flow_patch is not None:
                        flow_patches.append(flow_patch)

                in_batch = torch.cat(in_patches, dim=0)
                flow_batch = torch.cat(flow_patches, dim=0).to(self.device) if flow_patches else None
                if hasattr(self, 'netE'):
                    if flow_batch is not None:
                        out_batch = self.netE(in_batch, flow_spike=flow_batch).detach()
                    else:
                        out_batch = self.netE(in_batch).detach()
                else:
                    if flow_batch is not None:
                        out_batch = self.netG(in_batch, flow_spike=flow_batch).detach()
                    else:
                        out_batch = self.netG(in_batch).detach()

                if E is None:
                    c_out = out_batch.size(2)
                    E = torch.zeros(b, d, c_out, h*sf, w*sf, device=out_batch.device, dtype=out_batch.dtype)
                    W = torch.zeros_like(E)
                    print(
                        '[VAL_MODEL] _test_clip allocated '
                        f'E={tuple(E.shape)} ({_tensor_gb(E):.2f} GB) '
                        f'W={tuple(W.shape)} ({_tensor_gb(W):.2f} GB)'
                    )

                for patch_idx, (h_idx, w_idx) in enumerate(coord_batch):
                    out_patch = out_batch[patch_idx:patch_idx + 1]
                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size//2:, :] *= 0
                            out_patch_mask[..., -overlap_size//2:, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size//2:] *= 0
                            out_patch_mask[..., :, -overlap_size//2:] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., :overlap_size//2, :] *= 0
                            out_patch_mask[..., :overlap_size//2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, :overlap_size//2] *= 0
                            out_patch_mask[..., :, :overlap_size//2] *= 0

                    E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
            output = E.div_(W).cpu()

        else:
            _, _, _, h_old, w_old = lq.size()
            h_pad = (h_old// window_size[1]+1)*window_size[1] - h_old
            w_pad = (w_old// window_size[2]+1)*window_size[2] - w_old

            lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
            lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)
            if flow_spike is not None:
                flow_spike = torch.cat([flow_spike, torch.flip(flow_spike[:, :, :, -h_pad:, :], [3])], 3)
                flow_spike = torch.cat([flow_spike, torch.flip(flow_spike[:, :, :, :, -w_pad:], [4])], 4)
            if flow_spike_meta is not None:
                raise NotImplementedError("Lazy validation flow metadata requires size_patch_testing > 0.")

            if hasattr(self, 'netE'):
                if flow_spike is not None:
                    output = self.netE(lq, flow_spike=flow_spike).detach().cpu()
                else:
                    output = self.netE(lq).detach().cpu()
            else:
                if flow_spike is not None:
                    output = self.netG(lq, flow_spike=flow_spike).detach().cpu()
                else:
                    output = self.netG(lq).detach().cpu()

            output = output[:, :, :, :h_old*sf, :w_old*sf]

        return output

    def _normalize_flow_spike_meta(self, raw_meta):
        def _first(value):
            if isinstance(value, (list, tuple)):
                return _first(value[0])
            return value

        meta = {}
        for key, value in raw_meta.items():
            if key == 'frame_names':
                if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
                    value = [item[0] for item in value]
                meta[key] = [str(item) for item in value]
            elif key in {'subframes', 'source_h', 'source_w', 'target_h', 'target_w'}:
                meta[key] = int(_first(value))
            else:
                meta[key] = str(_first(value))
        return meta

    def _load_lazy_flow_patch(
        self,
        meta,
        temporal_offset,
        clip_len,
        h_idx,
        w_idx,
        patch_h,
        patch_w,
        full_h,
        full_w,
    ):
        src_h = int(meta['source_h'])
        src_w = int(meta['source_w'])
        src_top = round(h_idx * src_h / full_h)
        src_left = round(w_idx * src_w / full_w)
        src_crop_h = max(round(patch_h * src_h / full_h), 1)
        src_crop_w = max(round(patch_w * src_w / full_w), 1)
        src_top = max(min(src_top, src_h - src_crop_h), 0)
        src_left = max(min(src_left, src_w - src_crop_w), 0)

        flow_tensors = []
        frame_names = meta['frame_names'][temporal_offset:temporal_offset + clip_len]
        for frame_name in frame_names:
            base_path = Path(meta['flow_clip_dir']) / frame_name
            arr = load_encoding25_artifact_with_shape(
                base_path,
                artifact_format=meta['format'],
                num_subframes=int(meta['subframes']),
                spike_h=src_h,
                spike_w=src_w,
            )
            if int(meta['subframes']) > 1:
                for sub_idx in range(int(meta['subframes'])):
                    flow_tensors.append(
                        torch.from_numpy(self._crop_resize_flow_chw(arr[sub_idx], src_top, src_left, src_crop_h, src_crop_w, patch_h, patch_w))
                    )
            else:
                flow_tensors.append(
                    torch.from_numpy(self._crop_resize_flow_chw(arr, src_top, src_left, src_crop_h, src_crop_w, patch_h, patch_w))
                )
        flow_patch = torch.stack(flow_tensors, dim=0).unsqueeze(0).to(self.device)
        return flow_patch.float()

    def _load_lazy_flow_clip(self, meta, temporal_offset, clip_len):
        src_h = int(meta['source_h'])
        src_w = int(meta['source_w'])
        flow_tensors = []
        frame_names = meta['frame_names'][temporal_offset:temporal_offset + clip_len]
        for frame_name in frame_names:
            base_path = Path(meta['flow_clip_dir']) / frame_name
            arr = load_encoding25_artifact_with_shape(
                base_path,
                artifact_format=meta['format'],
                num_subframes=int(meta['subframes']),
                spike_h=src_h,
                spike_w=src_w,
            )
            if int(meta['subframes']) > 1:
                for sub_idx in range(int(meta['subframes'])):
                    flow_tensors.append(torch.from_numpy(np.asarray(arr[sub_idx], dtype=np.float32)))
            else:
                flow_tensors.append(torch.from_numpy(np.asarray(arr, dtype=np.float32)))
        return torch.stack(flow_tensors, dim=0)

    def _crop_resize_lazy_flow_patch(self, flow_clip, h_idx, w_idx, patch_h, patch_w, full_h, full_w):
        src_h = flow_clip.size(-2)
        src_w = flow_clip.size(-1)
        src_top = round(h_idx * src_h / full_h)
        src_left = round(w_idx * src_w / full_w)
        src_crop_h = max(round(patch_h * src_h / full_h), 1)
        src_crop_w = max(round(patch_w * src_w / full_w), 1)
        src_top = max(min(src_top, src_h - src_crop_h), 0)
        src_left = max(min(src_left, src_w - src_crop_w), 0)
        cropped = flow_clip[..., src_top:src_top + src_crop_h, src_left:src_left + src_crop_w]
        resized = torch.nn.functional.interpolate(
            cropped,
            size=(patch_h, patch_w),
            mode='bilinear',
            align_corners=False,
        )
        return resized.unsqueeze(0).to(self.device).float()

    def _crop_resize_lazy_flow_patch_cpu(self, flow_clip, h_idx, w_idx, patch_h, patch_w, full_h, full_w):
        src_h = flow_clip.size(-2)
        src_w = flow_clip.size(-1)
        src_top = round(h_idx * src_h / full_h)
        src_left = round(w_idx * src_w / full_w)
        src_crop_h = max(round(patch_h * src_h / full_h), 1)
        src_crop_w = max(round(patch_w * src_w / full_w), 1)
        src_top = max(min(src_top, src_h - src_crop_h), 0)
        src_left = max(min(src_left, src_w - src_crop_w), 0)
        cropped = flow_clip[..., src_top:src_top + src_crop_h, src_left:src_left + src_crop_w]
        resized = torch.nn.functional.interpolate(
            cropped,
            size=(patch_h, patch_w),
            mode='bilinear',
            align_corners=False,
        )
        return resized.unsqueeze(0).float()

    @staticmethod
    def _crop_resize_flow_chw(arr_chw, src_top, src_left, src_crop_h, src_crop_w, patch_h, patch_w):
        arr_chw = np.asarray(arr_chw)
        cropped = arr_chw[:, src_top:src_top + src_crop_h, src_left:src_left + src_crop_w]
        resized = [
            cv2.resize(cropped[ch], (patch_w, patch_h), interpolation=cv2.INTER_LINEAR)
            for ch in range(cropped.shape[0])
        ]
        return np.stack(resized, axis=0).astype(np.float32)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        self._print_different_keys_loading(network, state_dict, strict)
        network.load_state_dict(state_dict, strict=strict)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f'  {v}')
            print('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)
