import os
import shutil
import logging
import torch
import torch.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.utils_dist import get_local_rank, is_main_process


class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        
        # Device selection: use LOCAL_RANK in distributed mode, else cuda:0
        if opt.get('dist', False):
            local_rank = get_local_rank()
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers
        self._amp_dtypes = {
            'float16': torch.float16,
            'fp16': torch.float16,
            'half': torch.float16,
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
        }

    def _resolve_amp_dtype(self, raw_dtype, default='float16'):
        key = str(raw_dtype or default).strip().lower()
        if key not in self._amp_dtypes:
            raise ValueError(
                f'Unsupported AMP dtype {raw_dtype!r}. '
                f'Expected one of {sorted(self._amp_dtypes.keys())}.'
            )
        return self._amp_dtypes[key]

    def _autocast_context(self, enabled=False, dtype=torch.float16):
        enabled = bool(enabled) and self.device.type == 'cuda'
        return torch.amp.autocast(device_type='cuda', enabled=enabled, dtype=dtype)

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    def current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        """Get bare model under DDP/DataParallel and torch.compile wrappers."""
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        if hasattr(network, '_orig_mod'):
            network = network._orig_mod
        return network

    def _compile_options(self):
        train_opt = self.opt.get('train', {}) or {}
        compile_opt = train_opt.get('compile', {}) or {}
        if not isinstance(compile_opt, dict):
            raise ValueError("train.compile must be a dict when provided.")
        return compile_opt

    def _compile_scope(self, compile_opt):
        scope = str(compile_opt.get('scope', 'full_model')).strip().lower()
        if scope not in {'full_model', 'fusion_only'}:
            raise ValueError(
                f"Unsupported train.compile.scope={scope!r}; expected 'full_model' or 'fusion_only'."
            )
        return scope

    def _compile_kwargs(self, compile_opt):
        return {
            'mode': compile_opt.get('mode', 'default'),
            'fullgraph': bool(compile_opt.get('fullgraph', False)),
            'dynamic': bool(compile_opt.get('dynamic', True)),
            'backend': compile_opt.get('backend', 'inductor'),
        }

    def _compile_module(self, module, compile_opt, label, forward_only=False):
        if not hasattr(torch, 'compile'):
            if bool(compile_opt.get('fallback_on_error', True)):
                if self.opt.get('rank', 0) == 0:
                    logging.getLogger('train').warning(
                        '[COMPILE] torch.compile unavailable; using eager %s.', label
                    )
                return module
            raise RuntimeError('torch.compile is unavailable in this PyTorch build.')

        kwargs = self._compile_kwargs(compile_opt)
        target = module.forward if forward_only else module
        try:
            compiled = torch.compile(target, **kwargs)
            if self.opt.get('rank', 0) == 0:
                logging.getLogger('train').info('[COMPILE] Enabled torch.compile for %s with %s', label, kwargs)
            if forward_only:
                module.forward = compiled
                return module
            return compiled
        except Exception as exc:
            if bool(compile_opt.get('fallback_on_error', True)):
                if self.opt.get('rank', 0) == 0:
                    logging.getLogger('train').warning(
                        '[COMPILE] torch.compile failed for %s (%s); using eager module.', label, exc
                    )
                return module
            raise

    def compile_model_if_enabled(self, network):
        compile_opt = self._compile_options()
        if not bool(compile_opt.get('enable', False)):
            return network
        scope = self._compile_scope(compile_opt)
        if scope != 'full_model':
            if self.opt.get('rank', 0) == 0:
                logging.getLogger('train').info('[COMPILE] scope=%s; keeping netG eager.', scope)
            return network
        return self._compile_module(network, compile_opt, 'netG')

    def _fusion_compile_targets(self, network):
        targets = []
        adapter = getattr(network, 'fusion_adapter', None)
        operator = getattr(network, 'fusion_operator', None)
        if isinstance(operator, nn.Module):
            targets.append(('fusion_operator', network, 'fusion_operator', operator))
        if adapter is not None:
            for attr in ('operator',):
                module = getattr(adapter, attr, None)
                if isinstance(module, nn.Module):
                    targets.append((f'fusion_adapter.{attr}', adapter, attr, module))
            for adapter_attr in ('early_adapter', 'middle_adapter'):
                nested = getattr(adapter, adapter_attr, None)
                module = getattr(nested, 'operator', None)
                if isinstance(module, nn.Module):
                    targets.append((f'fusion_adapter.{adapter_attr}.operator', nested, 'operator', module))
        return targets

    def compile_fusion_modules_if_enabled(self, network):
        compile_opt = self._compile_options()
        scope = self._compile_scope(compile_opt)
        if not bool(compile_opt.get('enable', False)) or scope != 'fusion_only':
            return {'scope': scope, 'compiled': []}

        compiled = []
        compiled_by_id = {}
        for label, owner, attr, module in self._fusion_compile_targets(network):
            module_id = id(module)
            if module_id in compiled_by_id:
                setattr(owner, attr, compiled_by_id[module_id])
                continue
            original_forward = module.forward
            compiled_module = self._compile_module(module, compile_opt, label, forward_only=True)
            compiled_by_id[module_id] = compiled_module
            if compiled_module.forward is not original_forward:
                compiled.append(label)

        summary = {'scope': scope, 'compiled': compiled}
        if self.opt.get('rank', 0) == 0:
            logging.getLogger('train').info('[COMPILE] fusion scope summary: %s', summary)
        return summary

    def model_to_device(self, network):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)
        network = self.compile_model_if_enabled(network)
        if self.opt['dist']:
            # Use LOCAL_RANK for device assignment in DDP
            local_rank = get_local_rank()
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            use_static_graph = self.opt.get('use_static_graph', False)
            
            network = DistributedDataParallel(
                network, 
                device_ids=[local_rank], 
                output_device=local_rank,
                broadcast_buffers=False,  # Better performance, set to True if needed
                find_unused_parameters=find_unused_parameters
            )
            
            if use_static_graph:
                print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                network._set_static_graph()
        else:
            network = DataParallel(network)
        return network

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        # Only save on rank 0 in distributed mode
        if not is_main_process():
            return
            
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        tmp_save_path = save_path + '.tmp'
        
        try:
            network = self.get_bare_model(network)
            state_dict = network.state_dict()
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
            
            # Save to temporary file first (atomic write)
            torch.save(state_dict, tmp_save_path)
            
            # Atomically rename temporary file to final path
            # This ensures the checkpoint file is either complete or doesn't exist
            shutil.move(tmp_save_path, save_path)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(tmp_save_path):
                try:
                    os.remove(tmp_save_path)
                except OSError:
                    pass  # Ignore cleanup errors
            # Re-raise the exception to notify caller
            raise RuntimeError(f'Failed to save network checkpoint to {save_path}: {str(e)}') from e

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    def load_network_partial(self, load_path, network, param_key='params'):
        network = self.get_bare_model(network)
        state_dict_old = torch.load(load_path)
        if param_key in state_dict_old.keys():
            state_dict_old = state_dict_old[param_key]

        state_dict = network.state_dict()
        matched = {}
        skipped = []
        for key_old, param_old in state_dict_old.items():
            if key_old in state_dict and state_dict[key_old].shape == param_old.shape:
                matched[key_old] = param_old
            else:
                skipped.append(key_old)

        state_dict.update(matched)
        network.load_state_dict(state_dict, strict=True)
        print(
            f"Partial load: matched {len(matched)} tensors, skipped {len(skipped)} tensors from {load_path}"
        )

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        # Only save on rank 0 in distributed mode
        if not is_main_process():
            return
            
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        tmp_save_path = save_path + '.tmp'
        
        try:
            # Save to temporary file first (atomic write)
            torch.save(optimizer.state_dict(), tmp_save_path)
            
            # Atomically rename temporary file to final path
            # This ensures the checkpoint file is either complete or doesn't exist
            shutil.move(tmp_save_path, save_path)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(tmp_save_path):
                try:
                    os.remove(tmp_save_path)
                except OSError:
                    pass  # Ignore cleanup errors
            # Re-raise the exception to notify caller
            raise RuntimeError(f'Failed to save optimizer checkpoint to {save_path}: {str(e)}') from e

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
