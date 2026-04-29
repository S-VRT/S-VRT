from collections import OrderedDict
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip
from utils.utils_timer import Timer


_TRAINABLE_NAME_MARKERS = ("fusion_adapter", "fusion_operator", "lora_A", "lora_B")
_KNOWN_UNUSED_NAME_MARKERS = (
    "fusion_adapter.spike_upsample.refine",
    "fusion_adapter.early_adapter.spike_upsample.refine",
    "pa_deform.dcn.offset_mask",
)


def freeze_backbone(model):
    """Freeze backbone params; keep fusion adapters and LoRA adapters trainable."""
    for name, param in model.named_parameters():
        if any(marker in name for marker in _TRAINABLE_NAME_MARKERS):
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_known_unused_parameters(model):
    """Freeze parameters that are registered but bypassed by the current graph."""
    frozen = 0
    if not hasattr(model, 'named_parameters'):
        return frozen
    for name, param in model.named_parameters():
        if any(marker in name for marker in _KNOWN_UNUSED_NAME_MARKERS):
            if param.requires_grad:
                frozen += 1
            param.requires_grad = False
    return frozen


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.netG.to(self.device)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
        self.L_flow_spike = None
        amp_train_opt = self.opt_train.get('amp', {})
        self.amp_train_enabled = bool(amp_train_opt.get('enable', False)) and self.device.type == 'cuda'
        self.amp_train_dtype = self._resolve_effective_amp_dtype(amp_train_opt, phase='train')
        scaler_enabled = self.amp_train_enabled and self.amp_train_dtype == torch.float16
        self.grad_scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
        amp_val_opt = self.opt.get('val', {}).get('amp', {})
        self.amp_val_enabled = bool(amp_val_opt.get('enable', False)) and self.device.type == 'cuda'
        self.amp_val_dtype = self._resolve_effective_amp_dtype(
            amp_val_opt,
            phase='val',
            default=amp_train_opt.get('dtype', 'float16'),
        )
        self.batch_folder = None
        self.batch_lq_paths = None
        self.batch_gt_paths = None

    def _uses_dcnv4(self):
        dcn_type = self.opt.get('netG', {}).get('dcn_type', '')
        return str(dcn_type).strip().lower() == 'dcnv4'

    def _resolve_effective_amp_dtype(self, amp_opt, phase, default='float16'):
        raw_dtype = amp_opt.get('dtype', default)
        dtype = self._resolve_amp_dtype(raw_dtype, default=default)
        if self.device.type == 'cuda' and dtype == torch.bfloat16 and self._uses_dcnv4():
            print(
                f"[AMP] {phase} requested dtype={raw_dtype}, but DCNv4 backward does not support bfloat16. "
                "Falling back to float16."
            )
            amp_opt['dtype'] = 'float16'
            return torch.float16
        return dtype

    def _assert_lq_channels(self, tensor, tensor_name):
        """Validate the raw model-ingress tensor against configured input width."""
        _, _, expected_in_chans = self._resolve_input_config()
        actual_in_chans = tensor.size(2)
        if actual_in_chans != expected_in_chans:
            raise ValueError(
                f"{tensor_name} Channel Mismatch!\n"
                f"Tensor shape: {tensor.shape} (Channels: {actual_in_chans})\n"
                f"Expected raw ingress chans: {expected_in_chans}\n"
                f"Mode: {'Train' if self.netG.training else 'Test/Val'}\n"
                f"Hint: raw ingress width validates the tensor before concat/fusion handling "
                f"(e.g., RGB 3 + spike bins 8 = 11)."
            )


    def _flow_module_name(self):
        module = str(self.opt.get('netG', {}).get('optical_flow', {}).get('module', 'spynet')).strip().lower()
        if module == 'spike_flow':
            return 'scflow'
        return module

    def _resolve_input_config(self):
        net_cfg = self.opt.get('netG', {})
        input_cfg = net_cfg.get('input', {}) if isinstance(net_cfg.get('input', {}), dict) else {}
        fusion_cfg = net_cfg.get('fusion', {}) if isinstance(net_cfg.get('fusion', {}), dict) else {}
        raw_strategy = input_cfg.get(
            'strategy',
            'fusion' if fusion_cfg.get('enable', False) else 'concat',
        )
        strategy = str(raw_strategy).strip().lower()
        if strategy not in {'concat', 'fusion'}:
            raise ValueError(
                f"Unsupported netG.input.strategy={raw_strategy!r}; expected 'concat' or 'fusion'."
            )

        raw_mode = input_cfg.get('mode', net_cfg.get('input_mode', 'concat'))
        mode = str(raw_mode).strip().lower()
        if mode not in {'concat', 'dual'}:
            raise ValueError(
                f"Unsupported netG.input.mode={raw_mode!r}; expected 'concat' or 'dual'."
            )

        if strategy == 'fusion' and mode != 'dual':
            raise ValueError("input.strategy=fusion requires input.mode='dual'.")

        raw_ingress_chans = int(input_cfg.get('raw_ingress_chans', net_cfg.get('in_chans', 3)))
        return strategy, mode, raw_ingress_chans

    def _resolve_input_mode(self):
        _, mode, _ = self._resolve_input_config()
        return mode

    def _build_model_input_tensor(self, data):
        mode = self._resolve_input_mode()
        self._mark_net_input_path('concat_path' if mode == 'concat' else 'dual_path')
        if mode == 'concat':
            if 'L' not in data:
                raise KeyError("input_mode=concat requires data['L'].")
            return data['L']

        has_rgb = 'L_rgb' in data
        has_spike = 'L_spike' in data
        has_dual = has_rgb and has_spike
        if has_dual:
            l_rgb = data['L_rgb']
            l_spike = data['L_spike']
            self._validate_dual_input_tensors(l_rgb, l_spike)
            if l_rgb.shape[-2:] != l_spike.shape[-2:]:
                bsz, steps, spike_chans, _, _ = l_spike.shape
                target_h, target_w = l_rgb.shape[-2:]
                l_spike = F.interpolate(
                    l_spike.reshape(bsz * steps, spike_chans, l_spike.size(-2), l_spike.size(-1)),
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False,
                ).reshape(bsz, steps, spike_chans, target_h, target_w)
            return torch.cat([l_rgb, l_spike], dim=2)
        if has_rgb != has_spike:
            missing_key = "L_spike" if has_rgb else "L_rgb"
            raise KeyError(
                f"input_mode=dual received partial dual payload; missing data['{missing_key}']."
            )
        if 'L' in data:
            self._mark_net_input_path('dual_fallback_to_concat_path')
            print("[ModelPlain] input_mode=dual but L_rgb/L_spike missing; fallback to legacy data['L'].")
            return data['L']
        raise KeyError(
            "input_mode=dual requires both data['L_rgb'] and data['L_spike'], "
            "or legacy fallback data['L']."
        )

    def _validate_dual_input_tensors(self, l_rgb, l_spike):
        if l_rgb.ndim != 5 or l_spike.ndim != 5:
            raise ValueError(
                "input_mode=dual expects L_rgb and L_spike shaped [B,T,C,H,W]. "
                f"Got L_rgb ndim={l_rgb.ndim}, L_spike ndim={l_spike.ndim}."
            )
        if l_rgb.size(2) != 3:
            raise ValueError(
                f"input_mode=dual expects L_rgb channels=3, got {l_rgb.size(2)}."
            )
        if (
            l_rgb.size(0) != l_spike.size(0)
            or l_rgb.size(1) != l_spike.size(1)
        ):
            raise ValueError(
                f"input_mode=dual requires matching [B,T] between L_rgb {tuple(l_rgb.shape)} "
                f"and L_spike {tuple(l_spike.shape)}."
            )

    def _mark_net_input_path(self, marker):
        net = self.get_bare_model(self.netG) if hasattr(self, 'netG') else None
        if net is not None:
            if hasattr(net, "set_input_path_marker"):
                net.set_input_path_marker(marker)
            else:
                setattr(net, "_input_path_marker", marker)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        train_opt = self.opt.get('train', {})
        bare_model = self.get_bare_model(self.netG)

        lora_injected = False
        structure_changed = False
        if self._should_inject_lora_before_load(train_opt):
            structure_changed = self._inject_lora_adapters(train_opt, bare_model)
            lora_injected = True

        self.load()                           # load model

        if train_opt.get('use_lora', False) and not lora_injected:
            structure_changed = self._inject_lora_adapters(train_opt, bare_model)

        if train_opt.get('freeze_backbone', False):
            freeze_backbone(bare_model)
            if train_opt.get('use_lora', False) and train_opt.get('phase2_lora_mode', False):
                for name, param in bare_model.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        param.requires_grad = False
                print(f'[Phase 1] LoRA params frozen until iter {train_opt.get("fix_iter", 0)}')
            frozen_count = sum(1 for p in bare_model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in bare_model.parameters() if p.requires_grad)
            print(f'[Stage A/C] Frozen {frozen_count} params, trainable {trainable_count} params')
        if train_opt.get('freeze_known_unused_parameters', True):
            known_unused_count = freeze_known_unused_parameters(bare_model)
            if known_unused_count:
                print(f'[DDP] Frozen {known_unused_count} known-unused parameter tensors.')
        self.compile_fusion_modules_if_enabled(bare_model)
        self._wrap_netG_after_train_setup(bare_model, structure_changed=structure_changed)
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log
        # Optional timing/profiler instrumentation is off by default for production training.
        timing_cfg = self.opt_train.get('timing', {}) or {}
        profiler_cfg = self.opt_train.get('profiler', {}) or {}
        timing_enabled = bool(timing_cfg.get('enable', False))
        sync_cuda = bool(timing_cfg.get('sync_cuda', self.opt_train.get('sync_cuda_timing', False)))
        record_ranges = bool(profiler_cfg.get('record_ranges', False))
        self.timer = Timer(
            device=self.device,
            sync_cuda=sync_cuda,
            enabled=timing_enabled,
            record_ranges=record_ranges,
        )
        # 将计时器传递给网络，以便网络内部模块可以使用
        if hasattr(self.get_bare_model(self.netG), 'set_timer'):
            self.get_bare_model(self.netG).set_timer(self.timer)

    def _inject_lora_adapters(self, train_opt, bare_model):
        from models.lora import inject_lora

        targets = train_opt.get('lora_target_modules', ['qkv', 'proj'])
        rank = int(train_opt.get('lora_rank', 8))
        alpha = float(train_opt.get('lora_alpha', 16))
        replaced = inject_lora(bare_model, targets, rank=rank, alpha=alpha)
        print(f'[Stage C] Injected LoRA(rank={rank}, alpha={alpha}) into '
              f'{len(replaced)} Linear layers; targets={targets}')
        if train_opt.get('E_decay', 0) > 0 and hasattr(self, 'netE'):
            bare_e = self.get_bare_model(self.netE)
            inject_lora(bare_e, targets, rank=rank, alpha=alpha)
        return bool(replaced)

    def _wrap_netG_after_train_setup(self, bare_model, structure_changed=False):
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            if structure_changed:
                self._rewrap_distributed_netG_after_structure_change(bare_model, reason='LoRA injection')
            return
        self.netG = self.model_to_device(bare_model)

    def _rewrap_distributed_netG_after_structure_change(self, bare_model, reason):
        if not self.opt.get('dist', False):
            return
        print(f'[DDP] Re-wrapping netG after {reason} so DDP registers final trainable parameters.')
        self.netG = self.model_to_device(bare_model)

    def _should_inject_lora_before_load(self, train_opt):
        if not train_opt.get('use_lora', False):
            return False
        path_opt = self.opt.get('path', {})
        for path_key, param_key in (('pretrained_netG', 'params'), ('pretrained_netE', 'params_ema')):
            load_path = path_opt.get(path_key)
            if load_path is not None and self._checkpoint_contains_lora(load_path, param_key=param_key):
                return True
        return False

    @staticmethod
    def _checkpoint_contains_lora(load_path, param_key='params'):
        state_dict = torch.load(load_path, map_location='cpu')
        if isinstance(state_dict, dict) and param_key in state_dict:
            state_dict = state_dict[param_key]
        if not isinstance(state_dict, dict):
            return False
        return any('.lora_A.' in key or '.lora_B.' in key for key in state_dict.keys())

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            use_partial = self.opt.get('train', {}).get('partial_load', False)
            if use_partial:
                self.load_network_partial(load_path_G, self.netG, param_key='params')
            else:
                self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
        if self.grad_scaler.is_enabled() and load_path_optimizerG is not None:
            import os
            scaler_path = load_path_optimizerG.replace('_optimizerG.pth', '_scaler.pth')
            if os.path.exists(scaler_path):
                self.grad_scaler.load_state_dict(torch.load(scaler_path, map_location='cpu'))
                print(f'Loading GradScaler state from [{scaler_path}]')

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.grad_scaler.is_enabled():
            import os
            scaler_path = os.path.join(self.save_dir, f'{iter_label}_scaler.pth')
            torch.save(self.grad_scaler.state_dict(), scaler_path)

    def save_merged(self, iter_label):
        """Export LoRA-merged checkpoint (`{iter}_G_merged.pth` / `{iter}_E_merged.pth`).

        Produces state_dicts that are structurally identical to the non-LoRA VRT,
        so non-LoRA code paths can load them directly.
        Rank-0 only under DDP; no-op when use_lora is disabled.
        """
        import os
        if not self.opt.get('train', {}).get('use_lora', False):
            return
        if self.opt.get('rank', 0) != 0:
            return
        from models.lora import merged_state_dict

        pairs = [(self.netG, 'G')]
        if self.opt_train.get('E_decay', 0) > 0 and hasattr(self, 'netE'):
            pairs.append((self.netE, 'E'))

        for net, tag in pairs:
            bare = self.get_bare_model(net)
            state_dict = merged_state_dict(bare)
            save_path = os.path.join(self.save_dir, f'{iter_label}_{tag}_merged.pth')
            tmp_path = save_path + '.tmp'
            torch.save(state_dict, tmp_path)
            os.replace(tmp_path, save_path)
            print(f'[Stage C] Saved merged ckpt -> {save_path}')

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for name, param in self.netG.named_parameters():
            G_optim_params.append(param)
            if not param.requires_grad:
                print(f'Params [{name}] are frozen at optimizer build time but kept for later unfreeze.')
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(
                G_optim_params,
                lr=self.opt_train['G_optimizer_lr'],
                betas=self.opt_train['G_optimizer_betas'],
                weight_decay=self.opt_train['G_optimizer_wd'],
            )
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def _is_phase1_step(self, current_step):
        return (
            current_step is not None
            and hasattr(self, 'fix_iter')
            and self.fix_iter > 0
            and current_step < self.fix_iter
        )

    def _fusion_warmup_cfg(self) -> dict:
        return self.opt_train.get("fusion_warmup", {}) or {}

    def _resolve_fusion_warmup_stage(self, current_step):
        if not self._is_phase1_step(current_step):
            return "full"
        head_only_iters = int(self._fusion_warmup_cfg().get("head_only_iters", 0))
        if current_step < head_only_iters:
            return "writeback_only"
        return "token_mixer"

    def _configure_fusion_warmup_trainability(self, current_step):
        vrt = self.get_bare_model(self.netG)
        if not getattr(vrt, "fusion_enabled", False):
            return "full"
        operator = getattr(getattr(vrt, "fusion_adapter", None), "operator", None)
        if operator is None or not hasattr(operator, "set_warmup_stage"):
            return "full"
        stage = self._resolve_fusion_warmup_stage(current_step)
        operator.set_warmup_stage(stage)
        return stage

    def _record_fusion_diagnostics_to_log(self, fusion_meta) -> None:
        if not isinstance(fusion_meta, dict):
            return
        if "warmup_stage" in fusion_meta:
            self.log_dict["fusion_warmup_stage"] = fusion_meta["warmup_stage"]
        for key in ("token_norm", "mamba_norm", "pase_norm", "body_norm", "delta_norm", "gate_mean", "effective_update_norm"):
            if key in fusion_meta:
                self.log_dict[f"fusion_{key}"] = float(fusion_meta[key])

    def feed_data(self, data, need_H=True, current_step=None):
        with self.timer.timer('data_load'):
            self.batch_folder = data.get('folder')
            self.batch_lq_paths = data.get('lq_path')
            self.batch_gt_paths = data.get('gt_path')
            self.L = self._build_model_input_tensor(data).to(self.device)
            self._assert_lq_channels(self.L, 'Training Feed Data')

            if self._flow_module_name() == 'scflow' and not self._is_phase1_step(current_step):
                if 'L_flow_spike' not in data:
                    raise ValueError(
                        "module=scflow requires data['L_flow_spike'] with shape [B,T,25,H,W] "
                        "where T = frames (subframes=1) or frames*subframes (subframes>1)"
                    )
                self.L_flow_spike = data['L_flow_spike'].to(self.device)
                if self.L_flow_spike.ndim != 5 or self.L_flow_spike.size(2) != 25:
                    raise ValueError(
                        f"module=scflow requires L_flow_spike shape [B,T,25,H,W], got {tuple(self.L_flow_spike.shape)}"
                    )
            else:
                self.L_flow_spike = None

            if need_H:
                self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        with self.timer.timer('forward'):
            amp_enabled = getattr(self, 'amp_train_enabled', False)
            amp_dtype = getattr(self, 'amp_train_dtype', torch.float16)
            with self._autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                if self.L_flow_spike is not None:
                    self.E = self.netG(self.L, flow_spike=self.L_flow_spike)
                else:
                    self.E = self.netG(self.L)

    # ----------------------------------------
    # compute fusion auxiliary loss
    # ----------------------------------------
    def _resolve_phase1_passthrough_weight(self, current_step) -> float:
        warmup_cfg = self._fusion_warmup_cfg()
        start = float(warmup_cfg.get("passthrough_weight_start", self.opt_train.get("fusion_passthrough_loss_weight", 0.0)))
        end = float(warmup_cfg.get("passthrough_weight_end", self.opt_train.get("fusion_passthrough_loss_weight", 0.0)))
        if not hasattr(self, "fix_iter") or self.fix_iter <= 1:
            return end
        clamped = min(max(int(current_step or 0), 0), self.fix_iter - 1)
        progress = clamped / float(self.fix_iter - 1)
        return start + (end - start) * progress

    def _compute_fusion_aux_loss(self, is_phase1: bool, current_step=None) -> torch.Tensor:
        if is_phase1:
            aux_weight = self.opt_train.get('phase1_fusion_aux_loss_weight', 0.0)
            pass_weight = self._resolve_phase1_passthrough_weight(current_step)
            update_penalty_weight = float(self._fusion_warmup_cfg().get("update_penalty_weight", 0.0))
        else:
            aux_weight = self.opt_train.get('phase2_fusion_aux_loss_weight', 0.0)
            pass_weight = 0.0
            update_penalty_weight = 0.0
        if aux_weight == 0.0 and pass_weight == 0.0 and update_penalty_weight == 0.0:
            return torch.tensor(0.0, device=self.device)
        vrt = self.get_bare_model(self.netG)
        fusion_main = getattr(vrt, "_last_fusion_main", None)
        if fusion_main is None:
            return torch.tensor(0.0, device=self.device)
        if fusion_main.size(1) != self.H.size(1):
            raise ValueError(
                "Fusion aux loss expected canonical main timeline "
                f"N={self.H.size(1)}, got {fusion_main.size(1)}."
            )
        fusion_center = fusion_main
        loss = torch.tensor(0.0, device=self.device)
        if aux_weight > 0.0:
            loss = loss + aux_weight * self.G_lossfn(fusion_center, self.H)
        if pass_weight > 0.0:
            blur_rgb = self.L[:, :, :3, :, :]
            loss = loss + pass_weight * self.G_lossfn(fusion_center, blur_rgb)
        if update_penalty_weight > 0.0:
            effective_update = fusion_center - self.L[:, :, :3, :, :]
            loss = loss + update_penalty_weight * effective_update.abs().mean()
        return loss

    def _has_nonfinite_loss(self, loss: torch.Tensor) -> bool:
        is_nonfinite = (~torch.isfinite(loss.detach())).to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(is_nonfinite, op=dist.ReduceOp.MAX)
        return bool(is_nonfinite.item())

    # ----------------------------------------
    # phase1 fast-path: only run fusion_adapter, skip full VRT forward
    # ----------------------------------------
    def _phase1_fusion_forward(self):
        vrt = self.get_bare_model(self.netG)
        fusion_placement = vrt.fusion_cfg.get('placement', 'early') if vrt.fusion_enabled else None
        if not vrt.fusion_enabled or fusion_placement not in {'early', 'hybrid'}:
            self.netG_forward()
            return
        with self.timer.timer('forward'):
            with self._autocast_context(enabled=self.amp_train_enabled, dtype=self.amp_train_dtype):
                rgb = self.L[:, :, :3, :, :]
                spike = self.L[:, :, 3:, :, :]
                fusion_result = vrt.fusion_adapter(rgb=rgb, spike=spike)
                vrt._last_fusion_main = fusion_result["fused_main"]
                vrt._last_fusion_exec = fusion_result["backbone_view"]
                vrt._last_fusion_aux = fusion_result["aux_view"]
                vrt._last_fusion_meta = fusion_result["meta"]
                vrt._last_spike_bins = spike.shape[2]

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self._trace_current_step = current_step

        self._configure_fusion_warmup_trainability(current_step)

        is_phase1 = (
            hasattr(self, 'fix_iter')
            and self.fix_iter > 0
            and current_step < self.fix_iter
        )

        with self.timer.timer('zero_grad'):
            self.G_optimizer.zero_grad()

        if is_phase1:
            self._phase1_fusion_forward()
        else:
            self.netG_forward()

        with self.timer.timer('loss_compute'):
            if is_phase1:
                G_loss = self._compute_fusion_aux_loss(is_phase1=True, current_step=current_step)
            else:
                G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                G_loss = G_loss + self._compute_fusion_aux_loss(is_phase1=False, current_step=current_step)

        if self._has_nonfinite_loss(G_loss):
            self.log_dict['G_loss'] = G_loss.detach().item()
            self.log_dict['nan_guard_skip'] = 1.0
            self.log_dict['nan_guard_step'] = float(current_step)
            return

        with self.timer.timer('backward'):
            if self.grad_scaler.is_enabled():
                self.grad_scaler.scale(G_loss).backward()
            else:
                G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            with self.timer.timer('clip_grad'):
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.unscale_(self.G_optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        with self.timer.timer('optimizer_step'):
            if self.grad_scaler.is_enabled():
                self.grad_scaler.step(self.G_optimizer)
                self.grad_scaler.update()
            else:
                self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            with self.timer.timer('regularizer_orth'):
                self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            with self.timer.timer('regularizer_clip'):
                self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
        self.log_dict['nan_guard_skip'] = 0.0

        vrt = self.get_bare_model(self.netG)
        self._record_fusion_diagnostics_to_log(getattr(vrt, "_last_fusion_meta", None))

        if self.opt_train['E_decay'] > 0:
            with self.timer.timer('update_E'):
                self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        log_dict = self.log_dict.copy()
        # 添加耗时信息到日志
        current_timings = self.timer.get_current_timings()
        for key, value in current_timings.items():
            log_dict[f'time_{key}'] = value
        return log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
