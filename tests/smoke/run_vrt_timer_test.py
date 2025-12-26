import os
import torch

from models.architectures.vrt.vrt import VRT
from utils.utils_timer import Timer
from utils.utils_logger import Logger

def main():
    device = torch.device('cpu')

    # small config
    opt = {'path': {'log': './logs', 'tensorboard': './tb'}, 'task': 'smoke_test', 'logging': {}}

    # instantiate model (pa_frames=0 to avoid SpyNet/flow overhead)
    # use default img_size to match original window settings
    model = VRT(pa_frames=0, upscale=1)
    model.eval()

    # create timer and logger, inject timer to both
    timer = Timer(device=None, sync_cuda=False)
    model.set_timer(timer)
    logger = Logger(opt, logger=None, timer=timer)

    # random input (N, D, C, H, W)
    x = torch.rand(1, 6, 3, 64, 64)

    with torch.no_grad():
        out = model(x)

    print('output shape:', out.shape)

    # print timings via logger
    logger.log_timings(step=0)

if __name__ == '__main__':
    main()


