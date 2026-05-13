import os
from collections import OrderedDict
from datetime import datetime
import json
import re
import glob
import torch


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def get_timestamp():
    return datetime.now().strftime('_%y%m%d_%H%M%S')


def parse(opt_path, is_train=True):

    # ----------------------------------------
    # remove comments starting with '//'
    # ----------------------------------------
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

    # ----------------------------------------
    # initialize opt
    # ----------------------------------------
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['opt_path'] = opt_path
    opt['is_train'] = is_train

    # ----------------------------------------
    # set default
    # ----------------------------------------
    if 'merge_bn' not in opt:
        opt['merge_bn'] = False
        opt['merge_bn_startpoint'] = -1

    if 'scale' not in opt:
        opt['scale'] = 1

    # ----------------------------------------
    # default setting for logging
    # ----------------------------------------
    if 'logging' not in opt:
        opt['logging'] = {}
    if 'use_tensorboard' not in opt['logging']:
        opt['logging']['use_tensorboard'] = False
    if 'use_wandb' not in opt['logging']:
        opt['logging']['use_wandb'] = False
    if 'wandb_api_key' not in opt['logging']:
        opt['logging']['wandb_api_key'] = None
    if 'wandb_project' not in opt['logging']:
        opt['logging']['wandb_project'] = None
    if 'wandb_entity' not in opt['logging']:
        opt['logging']['wandb_entity'] = None
    if 'wandb_name' not in opt['logging']:
        opt['logging']['wandb_name'] = opt.get('task', 'experiment')
    if 'use_swanlab' not in opt['logging']:
        opt['logging']['use_swanlab'] = False
    if 'swanlab_api_key' not in opt['logging']:
        opt['logging']['swanlab_api_key'] = None
    if 'swanlab_project' not in opt['logging']:
        opt['logging']['swanlab_project'] = None
    if 'swanlab_workspace' not in opt['logging']:
        opt['logging']['swanlab_workspace'] = None
    if 'swanlab_name' not in opt['logging']:
        opt['logging']['swanlab_name'] = opt.get('task', 'experiment')
    if 'swanlab_description' not in opt['logging']:
        opt['logging']['swanlab_description'] = None
    if 'swanlab_mode' not in opt['logging']:
        opt['logging']['swanlab_mode'] = None
    if 'swanlab_auto_resume' not in opt['logging']:
        opt['logging']['swanlab_auto_resume'] = True
    if 'swanlab_resume_strategy' not in opt['logging']:
        opt['logging']['swanlab_resume_strategy'] = 'allow'
    if 'swanlab_run_id' not in opt['logging']:
        opt['logging']['swanlab_run_id'] = None
    if 'swanlab_run_id_file' not in opt['logging']:
        opt['logging']['swanlab_run_id_file'] = None
    if 'use_logfire' not in opt['logging']:
        opt['logging']['use_logfire'] = False
    if 'logfire_token' not in opt['logging']:
        opt['logging']['logfire_token'] = None
    if 'logfire_project_name' not in opt['logging']:
        opt['logging']['logfire_project_name'] = None
    if 'logfire_service_name' not in opt['logging']:
        opt['logging']['logfire_service_name'] = 's-vrt'
    if 'logfire_environment' not in opt['logging']:
        opt['logging']['logfire_environment'] = None
    if 'logfire_log_text' not in opt['logging']:
        opt['logging']['logfire_log_text'] = True
    if 'logfire_log_metrics' not in opt['logging']:
        opt['logging']['logfire_log_metrics'] = True
    if 'logfire_log_timings' not in opt['logging']:
        opt['logging']['logfire_log_timings'] = True

    # ----------------------------------------
    # datasets
    # ----------------------------------------
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = opt['scale']  # broadcast
        dataset['n_channels'] = opt['n_channels']  # broadcast
        if 'dataroot_H' in dataset and dataset['dataroot_H'] is not None:
            dataset['dataroot_H'] = os.path.expanduser(dataset['dataroot_H'])
        if 'dataroot_L' in dataset and dataset['dataroot_L'] is not None:
            dataset['dataroot_L'] = os.path.expanduser(dataset['dataroot_L'])

    # ----------------------------------------
    # path
    # ----------------------------------------
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)

    path_task = os.path.join(opt['path']['root'], opt['task'])
    opt['path']['task'] = path_task
    opt['path']['log'] = path_task
    opt['path']['options'] = os.path.join(path_task, 'options')

    if is_train:
        opt['path']['models'] = os.path.join(path_task, 'models')
        opt['path']['images'] = os.path.join(path_task, 'images')
        opt['path']['tensorboard'] = os.path.join(path_task, 'tensorboard')
    else:  # test
        opt['path']['images'] = os.path.join(path_task, 'test_images')

    # ----------------------------------------
    # network
    # ----------------------------------------
    opt['netG']['scale'] = opt['scale'] if 'scale' in opt else 1

    # ----------------------------------------
    # GPU devices and distributed training auto-detection
    # ----------------------------------------
    # Auto-detect distributed mode from WORLD_SIZE environment variable
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    
    if world_size > 1:
        # Distributed mode: do NOT set CUDA_VISIBLE_DEVICES
        # Platform DDP or torchrun has already set up device assignment
        opt['dist'] = True
        opt['num_gpu'] = world_size
        print(f'Distributed training detected: {world_size} GPUs')
        print('Note: gpu_ids in config is ignored in distributed mode')
    else:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible_devices is not None:
            visible_list = [x.strip() for x in visible_devices.split(',') if x.strip()]
            opt['num_gpu'] = len(visible_list)
            opt['gpu_ids'] = list(range(opt['num_gpu']))
            print('CUDA_VISIBLE_DEVICES already set: ' + (visible_devices if visible_devices else '<empty>'))
            print('Using logical gpu_ids from visible devices: ' + str(opt['gpu_ids']))
        else:
            detected_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if detected_gpu_count > 0:
                opt['num_gpu'] = detected_gpu_count
                opt['gpu_ids'] = list(range(detected_gpu_count))
                print('CUDA_VISIBLE_DEVICES not set; using all visible GPUs detected by PyTorch: ' + str(opt['gpu_ids']))
            elif 'gpu_ids' in opt and opt['gpu_ids']:
                opt['num_gpu'] = len(opt['gpu_ids'])
                print('CUDA_VISIBLE_DEVICES not set and no CUDA device detected; keeping config gpu_ids: ' + str(opt['gpu_ids']))
            else:
                opt['num_gpu'] = 0
                opt['gpu_ids'] = []
        opt['dist'] = False
        print('number of GPUs is: ' + str(opt['num_gpu']))

    # ----------------------------------------
    # default setting for distributeddataparallel
    # ----------------------------------------
    if 'find_unused_parameters' not in opt:
        opt['find_unused_parameters'] = False
    if 'use_static_graph' not in opt:
        opt['use_static_graph'] = False

    # ----------------------------------------
    # default setting for perceptual loss
    # ----------------------------------------
    if 'F_feature_layer' not in opt['train']:
        opt['train']['F_feature_layer'] = 34  # 25; [2,7,16,25,34]
    if 'F_weights' not in opt['train']:
        opt['train']['F_weights'] = 1.0  # 1.0; [0.1,0.1,1.0,1.0,1.0]
    if 'F_lossfn_type' not in opt['train']:
        opt['train']['F_lossfn_type'] = 'l1'
    if 'F_use_input_norm' not in opt['train']:
        opt['train']['F_use_input_norm'] = True
    if 'F_use_range_norm' not in opt['train']:
        opt['train']['F_use_range_norm'] = False

    # ----------------------------------------
    # default setting for optimizer
    # ----------------------------------------
    if 'G_optimizer_type' not in opt['train']:
        opt['train']['G_optimizer_type'] = "adam"
    if 'G_optimizer_betas' not in opt['train']:
        opt['train']['G_optimizer_betas'] = [0.9,0.999]
    if 'G_scheduler_restart_weights' not in opt['train']:
        opt['train']['G_scheduler_restart_weights'] = 1
    if 'G_optimizer_wd' not in opt['train']:
        opt['train']['G_optimizer_wd'] = 0
    if 'G_optimizer_reuse' not in opt['train']:
        opt['train']['G_optimizer_reuse'] = False
    if 'netD' in opt and 'D_optimizer_reuse' not in opt['train']:
        opt['train']['D_optimizer_reuse'] = False

    if 'two_run' not in opt['train']:
        opt['train']['two_run'] = {'enable': False}
    elif not isinstance(opt['train']['two_run'], dict):
        raise ValueError("train.two_run must be a dict when provided.")

    two_run_cfg = opt['train']['two_run']
    if 'enable' not in two_run_cfg:
        two_run_cfg['enable'] = False
    if two_run_cfg.get('enable', False):
        if 'phase1' not in two_run_cfg or 'phase2' not in two_run_cfg:
            raise ValueError("train.two_run.enable=true requires both phase1 and phase2 blocks.")

    # ----------------------------------------
    # default setting of strict for model loading
    # ----------------------------------------
    if 'G_param_strict' not in opt['train']:
        opt['train']['G_param_strict'] = True
    if 'netD' in opt and 'D_param_strict' not in opt['path']:
        opt['train']['D_param_strict'] = True
    if 'E_param_strict' not in opt['path']:
        opt['train']['E_param_strict'] = True

    # ----------------------------------------
    # Exponential Moving Average
    # ----------------------------------------
    if 'E_decay' not in opt['train']:
        opt['train']['E_decay'] = 0

    # ----------------------------------------
    # default setting for discriminator
    # ----------------------------------------
    if 'netD' in opt:
        if 'net_type' not in opt['netD']:
            opt['netD']['net_type'] = 'discriminator_patchgan'  # discriminator_unet
        if 'in_nc' not in opt['netD']:
            opt['netD']['in_nc'] = 3
        if 'base_nc' not in opt['netD']:
            opt['netD']['base_nc'] = 64
        if 'n_layers' not in opt['netD']:
            opt['netD']['n_layers'] = 3
        if 'norm_type' not in opt['netD']:
            opt['netD']['norm_type'] = 'spectral'


    return opt


def find_last_checkpoint(save_dir, net_type='G', pretrained_path=None):
    """
    Args: 
        save_dir: model folder
        net_type: 'G' or 'D' or 'optimizerG' or 'optimizerD'
        pretrained_path: pretrained model path. If save_dir does not have any model, load from pretrained_path

    Return:
        init_iter: iteration number
        init_path: model path
    """
    file_list = glob.glob(os.path.join(save_dir, '*_{}.pth'.format(net_type)))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, '{}_{}.pth'.format(init_iter, net_type))
    else:
        init_iter = 0
        init_path = pretrained_path
    return init_iter, init_path


'''
# --------------------------------------------
# convert the opt into json file
# --------------------------------------------
'''


def save(opt):
    opt_path = opt['opt_path']
    opt_path_copy = opt['path']['options']
    dirname, filename_ext = os.path.split(opt_path)
    filename, ext = os.path.splitext(filename_ext)
    dump_path = os.path.join(opt_path_copy, filename+get_timestamp()+ext)
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


'''
# --------------------------------------------
# dict to string for logger
# --------------------------------------------
'''


def dict2str(opt, indent_l=1):
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


'''
# --------------------------------------------
# convert OrderedDict to NoneDict,
# return None for missing key
# --------------------------------------------
'''


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None
