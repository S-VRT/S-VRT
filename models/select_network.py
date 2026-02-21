import functools
import os
import torch
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # VRT
    # ----------------------------------------
    if net_type == 'vrt':
        from models.architectures.vrt import VRT as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net.get('in_chans', 3),
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   depths=opt_net['depths'],
                   indep_reconsts=opt_net['indep_reconsts'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   optical_flow=opt_net['optical_flow'],
                   pa_frames=opt_net['pa_frames'],
                   deformable_groups=opt_net['deformable_groups'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'],
                   use_sgp=opt_net.get('use_sgp', False),
                   sgp_w=opt_net.get('sgp_w', 3),
                   sgp_k=opt_net.get('sgp_k', 3),
                   sgp_reduction=opt_net.get('sgp_reduction', 4),
                   use_flash_attn=opt_net.get('use_flash_attn', True),
                   dcn_config={
                       'type': opt_net['dcn_type'],
                       'apply_softmax': opt_net['dcn_apply_softmax']
                   },
                   opt=opt)

        # ----------------------------------------
        # RVRT
        # ----------------------------------------
    elif net_type == 'rvrt':
        from models.network_rvrt import RVRT as net
        netG = net(upscale=opt_net['upscale'],
                   clip_size=opt_net['clip_size'],
                   img_size=opt_net['img_size'],
                   window_size=opt_net['window_size'],
                   num_blocks=opt_net['num_blocks'],
                   depths=opt_net['depths'],
                   embed_dims=opt_net['embed_dims'],
                   num_heads=opt_net['num_heads'],
                   inputconv_groups=opt_net['inputconv_groups'],

                   deformable_groups=opt_net['deformable_groups'],
                   attention_heads=opt_net['attention_heads'],
                   attention_window=opt_net['attention_window'],
                   nonblind_denoising=opt_net['nonblind_denoising'],
                   use_checkpoint_attn=opt_net['use_checkpoint_attn'],
                   use_checkpoint_ffn=opt_net['use_checkpoint_ffn'],
                   no_checkpoint_attn_blocks=opt_net['no_checkpoint_attn_blocks'],
                   no_checkpoint_ffn_blocks=opt_net['no_checkpoint_ffn_blocks'],
                   cpu_cache_length=opt_net['cpu_cache_length'])

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])
    # ----------------------------------------
    # move model to device (centralized placement)
    # prefer LOCAL_RANK if present (DDP), else use first gpu_id from config, else cpu
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        if local_rank >= 0:
            device = torch.device(f'cuda:{local_rank}')
        else:
            gpu_ids = opt.get('gpu_ids', None)
            if gpu_ids and len(gpu_ids) > 0:
                device = torch.device(f'cuda:{gpu_ids[0]}')
            else:
                device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    netG = netG.to(device)

    return netG



"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
