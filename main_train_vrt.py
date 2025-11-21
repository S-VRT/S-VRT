"""
VRT/RVRT 模型训练脚本
用于训练视频恢复任务（如视频去模糊、视频超分辨率等）的 VRT (Video Restoration Transformer) 模型
"""

import sys  # 系统相关功能，用于程序退出
import os.path  # 路径操作
import math  # 数学运算，用于计算迭代次数等
import argparse  # 命令行参数解析
import time  # 时间相关功能
import random  # 随机数生成，用于设置随机种子
import cv2  # OpenCV，用于图像读写
import numpy as np  # 数值计算库
from collections import OrderedDict  # 有序字典，用于保持测试结果的顺序
import logging  # 日志记录
import torch  # PyTorch 深度学习框架
import torch.distributed as dist
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.data.distributed import DistributedSampler  # 分布式训练的数据采样器

# 工具函数导入
from utils import utils_logger  # 日志工具，包括 TensorBoard 和 WANDB 支持
from utils import utils_image as util  # 图像处理工具函数
from utils import utils_option as option  # 配置文件解析工具
from utils.utils_dist import get_dist_info, init_dist, barrier_safe, setup_distributed, get_rank, is_main_process  # 分布式训练工具

# 数据集和模型定义
from data.select_dataset import define_Dataset  # 数据集工厂函数
from models.select_model import define_Model  # 模型工厂函数


def main():
    """
    主训练函数，接收命令行参数指定的配置文件路径
    """
    
    '''
    # ----------------------------------------
    # 步骤 1: 准备配置选项 (prepare opt)
    # ----------------------------------------
    '''
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='配置文件 JSON 文件路径')
    # 以下参数为向后兼容保留，但会被自动检测忽略
    parser.add_argument('--launcher', default='pytorch', help='任务启动器（已忽略，自动检测）')
    parser.add_argument('--local_rank', type=int, default=0, help='本地进程排名（已忽略，使用环境变量）')
    parser.add_argument('--dist', default=False, help='分布式模式（已忽略，自动检测）')
    
    args = parser.parse_args()  # 解析命令行参数

    # ----------------------------------------
    # 首先初始化分布式训练环境
    # ----------------------------------------
    # 在命令行解析之后、加载 JSON 配置之前立即初始化，以确保正确的设备设置
    # 这会检测环境变量（如 WORLD_SIZE, RANK）并设置分布式训练
    setup_distributed()
    
    # ----------------------------------------
    # 解析配置文件并自动检测分布式模式
    # ----------------------------------------
    # 从 JSON 文件加载所有训练配置（学习率、批次大小、模型参数等）
    opt = option.parse(args.opt, is_train=True)
    # opt['dist'] 已由 option.parse() 根据 WORLD_SIZE 环境变量自动设置
    # 获取当前进程的排名和总进程数
    opt['rank'], opt['world_size'] = get_dist_info()
    
    # ----------------------------------------
    # 创建必要的目录（仅主进程 rank 0 执行）
    # ----------------------------------------
    # 在分布式训练中，只有主进程创建目录，避免多进程竞争
    if is_main_process():
        # 创建所有必要的目录（模型保存、日志、图像输出等），但不包括预训练模型路径
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # 更新配置选项：查找并加载检查点
    # ----------------------------------------
    # 查找最新的检查点文件，用于恢复训练或加载预训练模型
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    
    # 查找生成器网络 G 的最新检查点
    # init_iter_G: 检查点对应的迭代次数，init_path_G: 检查点文件路径
    # 如果 pretrained_netG 为 None/null，则从头开始训练，不自动查找检查点
    # 如果 pretrained_netG 为 "auto"，则自动查找检查点（用于恢复训练）
    # 如果 pretrained_netG 为其他字符串（指定了路径），则使用指定的路径，不自动查找检查点
    if opt['path']['pretrained_netG'] is None:
        init_iter_G = 0
        init_path_G = None
    elif opt['path']['pretrained_netG'] == "auto":
        # "auto" 表示自动查找检查点（恢复训练）
        init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',
                                                               pretrained_path=None)
    else:
        # 指定了路径，直接使用指定的路径
        init_iter_G = 0  # 无法从路径中提取迭代次数，设为 0
        init_path_G = opt['path']['pretrained_netG']
    
    # 查找编码器网络 E 的最新检查点
    # 如果 pretrained_netE 为 None/null，则不自动查找检查点
    # 如果 pretrained_netE 为 "auto"，则自动查找检查点（用于恢复训练）
    # 如果 pretrained_netE 为其他字符串（指定了路径），则使用指定的路径，不自动查找检查点
    if opt['path']['pretrained_netE'] is None:
        init_iter_E = 0
        init_path_E = None
    elif opt['path']['pretrained_netE'] == "auto":
        # "auto" 表示自动查找检查点（恢复训练）
        init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E',
                                                               pretrained_path=None)
    else:
        # 指定了路径，直接使用指定的路径
        init_iter_E = 0  # 无法从路径中提取迭代次数，设为 0
        init_path_E = opt['path']['pretrained_netE']
    
    # 更新配置中的预训练模型路径
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    
    # 查找优化器的最新检查点（只有当 G 或 E 有检查点时才查找优化器检查点）
    if init_iter_G > 0 or init_iter_E > 0:
        init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                                 net_type='optimizerG')
    else:
        init_iter_optimizerG = 0
        init_path_optimizerG = None
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    
    # 当前训练步数取所有检查点中最大的迭代次数（用于恢复训练）
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # 保存配置到 '../option.json' 文件
    # ----------------------------------------
    # 仅主进程保存，用于记录当前训练使用的配置
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # 将字典中缺失的键设置为 None
    # ----------------------------------------
    # 这样在访问不存在的键时不会报错，而是返回 None
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # 配置日志记录器
    # ----------------------------------------
    # 仅主进程初始化日志，避免多进程重复写入
    if opt['rank'] == 0:
        logger_name = 'train'
        # 设置日志文件路径
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        # 记录完整的配置信息到日志
        logger.info(option.dict2str(opt))
        
        # 初始化 TensorBoard 和 WANDB 日志记录器
        # 用于可视化训练过程（损失曲线、学习率等）
        tb_logger = utils_logger.Logger(opt, logger)
    else:
        # 非主进程不初始化日志记录器
        tb_logger = None

    # ----------------------------------------
    # 设置随机种子（分布式训练时根据 rank 偏移）
    # ----------------------------------------
    # 获取配置中的手动种子，如果未设置则随机生成
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    
    # 为每个进程添加 rank 偏移，使不同进程有不同的随机状态
    # 这对于数据增强的多样性很重要，避免所有进程看到相同的数据
    seed_rank = seed + opt['rank']
    
    # 打印种子信息
    if opt['rank'] == 0:
        print('Base random seed: {}'.format(seed))
    print('Rank {} using seed: {}'.format(opt['rank'], seed_rank))
    
    # 设置所有随机数生成器的种子，确保结果可复现
    random.seed(seed_rank)  # Python 随机数
    np.random.seed(seed_rank)  # NumPy 随机数
    torch.manual_seed(seed_rank)  # PyTorch CPU 随机数
    torch.cuda.manual_seed_all(seed_rank)  # PyTorch GPU 随机数（所有 GPU）

    '''
    # ----------------------------------------
    # 步骤 2: 创建数据加载器 (create dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) 创建数据集
    # 2) 为训练集和测试集创建数据加载器
    # ----------------------------------------
    # 遍历配置中的所有数据集（通常包括 'train' 和 'test'）
    test_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # 创建训练数据集
            train_set = define_Dataset(dataset_opt)
            # 计算训练迭代次数（向上取整）
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            
            if opt['dist']:
                # 分布式训练模式
                # 使用 DistributedSampler 确保每个进程看到不同的数据子集
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'],
                                                   drop_last=True, seed=seed)  # 使用基础种子，不是 seed_rank
                # 创建数据加载器
                # 批次大小需要除以 GPU 数量，因为每个 GPU 处理一部分数据
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,  # 分布式训练时由 sampler 控制打乱
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],  # 每个 GPU 的工作进程数
                                          drop_last=True,  # 丢弃最后一个不完整的批次
                                          pin_memory=True,  # 将数据固定在内存中，加速 GPU 传输
                                          sampler=train_sampler)  # 使用分布式采样器
            else:
                # 单 GPU 训练模式
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            # 创建测试/验证数据集
            test_set = define_Dataset(dataset_opt)
            # 允许通过配置指定 DataLoader 的各项参数
            test_batch_size = dataset_opt.get('dataloader_batch_size', 1)
            test_num_workers = dataset_opt.get('dataloader_num_workers', 1)
            test_shuffle = dataset_opt.get('dataloader_shuffle', False)

            if opt['dist']:
                # 分布式验证 / 测试
                test_sampler = DistributedSampler(test_set, shuffle=test_shuffle,
                                                  drop_last=False, seed=seed)
                test_loader = DataLoader(test_set,
                                         batch_size=max(1, test_batch_size // opt['num_gpu']),
                                         shuffle=False,
                                         num_workers=max(1, test_num_workers // opt['num_gpu']),
                                         drop_last=False,
                                         pin_memory=True,
                                         sampler=test_sampler)
            else:
                # 单卡验证 / 测试
                test_loader = DataLoader(test_set,
                                         batch_size=test_batch_size,
                                         shuffle=test_shuffle,
                                         num_workers=test_num_workers,
                                         drop_last=False,
                                         pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # 步骤 3: 初始化模型 (initialize model)
    # ----------------------------------------
    '''

    # 根据配置创建模型（VRT 或 RVRT）
    model = define_Model(opt)
    # 初始化训练相关组件（优化器、损失函数、学习率调度器等）
    model.init_train()
    if opt['rank'] == 0:
        # 记录网络结构和参数信息
        logger.info(model.info_network())  # 网络架构信息
        logger.info(model.info_params())  # 参数数量信息

    '''
    # ----------------------------------------
    # 步骤 4: 主训练循环 (main training)
    # ----------------------------------------
    '''

    # 开始训练循环（最多运行 1000000 个 epoch，实际由 total_iter 控制）
    for epoch in range(1000000):  # 持续运行直到达到总迭代次数
        # 为 DistributedSampler 设置 epoch，确保每个 epoch 数据打乱顺序不同
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        
        # 遍历训练数据
        for i, train_data in enumerate(train_loader):

            current_step += 1  # 更新当前训练步数

            # -------------------------------
            # 1) 更新学习率
            # -------------------------------
            # 根据当前步数调整学习率（可能使用学习率调度器）
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) 输入数据对（低质量图像和高质量图像）
            # -------------------------------
            # 将训练数据（低质量输入和高质量标签）送入模型
            model.feed_data(train_data)

            # -------------------------------
            # 3) 优化模型参数
            # -------------------------------
            # 执行前向传播、计算损失、反向传播和参数更新
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) 记录训练信息
            # -------------------------------
            # 每隔一定步数打印训练信息（损失值、学习率等）
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # 获取当前损失等信息（如 loss）
                # 构建日志消息：包含 epoch、迭代次数、学习率
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                # 将损失信息合并到消息中
                for k, v in logs.items():  # 合并日志信息到消息
                    if k.startswith('time_'):
                        # 耗时信息使用不同的格式
                        message += '{:s}: {:.4f}s '.format(k.replace('time_', ''), v)
                    else:
                        message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)  # 写入日志文件
                
                # 记录到 TensorBoard 和 WANDB（用于可视化）
                if tb_logger is not None:
                    # 分离训练指标和耗时指标，分别记录到不同命名空间
                    train_logs = {k: v for k, v in logs.items() if not k.startswith('time_')}
                    train_logs['learning_rate'] = model.current_learning_rate()
                    # 耗时指标去掉time_前缀，记录到time命名空间
                    time_logs = {k.replace('time_', ''): v for k, v in logs.items() if k.startswith('time_')}
                    
                    if train_logs:
                        tb_logger.log_scalars(current_step, train_logs, tag_prefix='train')
                    if time_logs:
                        tb_logger.log_scalars(current_step, time_logs, tag_prefix='time')

            # -------------------------------
            # 5) 保存模型检查点
            # -------------------------------
            # 每隔一定步数保存模型（包括网络权重和优化器状态）
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)  # 保存当前步数的模型
            # 在分布式训练中，等待 rank 0 完成保存，避免进程间状态不一致
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['dist']:
                barrier_safe()  # 同步所有进程

            # 特殊处理：当使用静态计算图时，在改变计算图之前提前保存模型
            # 这是因为在分布式训练中使用 use_checkpoint=True 时存在 bug
            if opt['use_static_graph'] and (current_step == opt['train']['fix_iter'] - 1):
                current_step += 1
                model.update_learning_rate(current_step)
                if opt['rank'] == 0:
                    model.save(current_step)  # 提前保存模型
                # 等待 rank 0 完成保存
                if opt['dist']:
                    barrier_safe()
                current_step -= 1  # 恢复步数
                if opt['rank'] == 0:
                    logger.info('Saving models ahead of time when changing the computation graph with use_static_graph=True'
                                ' (we need it due to a bug with use_checkpoint=True in distributed training). The training '
                                'will be terminated by PyTorch in the next iteration. Just resume training with the same '
                                '.json config file.')

            # -------------------------------
            # 6) 模型测试和评估
            # -------------------------------
            # 每隔一定步数在测试集上评估模型性能
            if current_step % opt['train']['checkpoint_test'] == 0:

                is_master_process = opt['rank'] == 0
                if opt['dist']:
                    barrier_safe()

                # 初始化测试结果字典，用于存储所有测试序列的指标
                test_results = OrderedDict()
                test_results['psnr'] = []  # 峰值信噪比（RGB 通道）
                test_results['ssim'] = []  # 结构相似性指数（RGB 通道）
                test_results['psnr_y'] = []  # 峰值信噪比（Y 通道，亮度）
                test_results['ssim_y'] = []  # 结构相似性指数（Y 通道，亮度）

                # 初始化 gt 变量，避免在循环外部引用时出现 UnboundLocalError
                gt = None

                # 遍历测试数据
                for idx, test_data in enumerate(test_loader):
                    # 将测试数据送入模型
                    model.feed_data(test_data)
                    # 执行测试（前向传播，不更新梯度）
                    model.test()

                    # 获取模型输出和真实标签
                    visuals = model.current_visuals()
                    output = visuals['E']  # E: 估计/输出图像 (Estimated)
                    gt = visuals['H'] if 'H' in visuals else None  # H: 高质量真实图像 (High-quality)
                    folder = test_data['folder']  # 测试序列的文件夹名称

                    # 初始化当前测试序列的结果字典
                    test_results_folder = OrderedDict()
                    test_results_folder['psnr'] = []
                    test_results_folder['ssim'] = []
                    test_results_folder['psnr_y'] = []
                    test_results_folder['ssim_y'] = []

                    # 处理批次中的每一张图像
                    for i in range(output.shape[0]):
                        # -----------------------
                        # 保存估计的图像 E
                        # -----------------------
                        # 将输出张量转换为 numpy 数组，并限制在 [0, 1] 范围内
                        img = output[i, ...].clamp_(0, 1).numpy()
                        if img.ndim == 3:
                            # 从 CHW-RGB 格式转换为 HWC-BGR 格式（OpenCV 使用 BGR）
                            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
                        # 将浮点数 [0, 1] 转换为 uint8 [0, 255]
                        img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                        
                        # 如果配置要求保存图像，则保存到磁盘
                        if opt['val']['save_img']:
                            save_dir = opt['path']['images']
                            util.mkdir(save_dir)
                            # 从输入路径提取序列名称
                            seq_ = os.path.basename(test_data['lq_path'][i][0]).split('.')[0]
                            # 创建文件夹并保存图像
                            os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                            cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}_{current_step:d}.png', img)

                        # -----------------------
                        # 计算 PSNR 和 SSIM
                        # -----------------------
                        # 只有在有真实标签时才计算指标
                        if gt is not None:
                            # 处理真实标签图像
                            img_gt = gt[i, ...].clamp_(0, 1).numpy()
                            if img_gt.ndim == 3:
                                # 从 CHW-RGB 格式转换为 HWC-BGR 格式
                                img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
                            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                            img_gt = np.squeeze(img_gt)  # 移除单维度

                            # 计算 RGB 通道的 PSNR 和 SSIM
                            test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                            test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                            
                            # 如果是 RGB 图像，计算 Y 通道（亮度）的指标
                            if img_gt.ndim == 3:  # RGB image
                                # 转换为 YCbCr 颜色空间，只取 Y 通道（亮度）
                                img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                                img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                                # 计算 Y 通道的 PSNR 和 SSIM
                                test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                                test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                            else:
                                # 灰度图像，Y 通道指标等于 RGB 指标
                                test_results_folder['psnr_y'] = test_results_folder['psnr']
                                test_results_folder['ssim_y'] = test_results_folder['ssim']

                    # 如果有计算的指标，记录并保存结果
                    if len(test_results_folder['psnr']) > 0:
                        # 计算当前测试序列的平均指标
                        psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                        ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                        psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
                        ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])

                        if is_master_process:
                            logger.info('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                        'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                                        format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
                        test_results['psnr'].append(psnr)
                        test_results['ssim'].append(ssim)
                        test_results['psnr_y'].append(psnr_y)
                        test_results['ssim_y'].append(ssim_y)
                    else:
                        # 没有真实标签时，只记录测试序列名称
                        if is_master_process:
                            logger.info('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

                local_psnr_sum = sum(test_results['psnr'])
                local_ssim_sum = sum(test_results['ssim'])
                local_psnr_y_sum = sum(test_results['psnr_y'])
                local_ssim_y_sum = sum(test_results['ssim_y'])

                local_psnr_count = len(test_results['psnr'])
                local_ssim_count = len(test_results['ssim'])
                local_psnr_y_count = len(test_results['psnr_y'])
                local_ssim_y_count = len(test_results['ssim_y'])

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                metrics_tensor = torch.tensor(
                    [local_psnr_sum, local_ssim_sum, local_psnr_y_sum, local_ssim_y_sum,
                     local_psnr_count, local_ssim_count, local_psnr_y_count, local_ssim_y_count],
                    dtype=torch.float64, device=device)

                if opt['dist']:
                    dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

                (global_psnr_sum, global_ssim_sum, global_psnr_y_sum, global_ssim_y_sum,
                 global_psnr_count, global_ssim_count, global_psnr_y_count,
                 global_ssim_y_count) = metrics_tensor.tolist()

                global_psnr_count = int(round(global_psnr_count))
                global_ssim_count = int(round(global_ssim_count))
                global_psnr_y_count = int(round(global_psnr_y_count))
                global_ssim_y_count = int(round(global_ssim_y_count))

                if global_psnr_count > 0:
                    ave_psnr = global_psnr_sum / global_psnr_count
                    ave_ssim = global_ssim_sum / max(global_ssim_count, 1)
                    ave_psnr_y = global_psnr_y_sum / max(global_psnr_y_count, 1)
                    ave_ssim_y = global_ssim_y_sum / max(global_ssim_y_count, 1)

                    if is_master_process:
                        logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                    'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                            epoch, current_step, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))
                        
                        # 将测试指标记录到 TensorBoard 和 WANDB
                        if tb_logger is not None:
                            test_metrics = {
                                'psnr': ave_psnr,
                                'ssim': ave_ssim,
                                'psnr_y': ave_psnr_y,
                                'ssim_y': ave_ssim_y
                            }
                            tb_logger.log_scalars(current_step, test_metrics, tag_prefix='test')

                if opt['dist']:
                    barrier_safe()

            # 检查是否达到总迭代次数，如果达到则结束训练
            if current_step > opt['train']['total_iter']:
                logger.info('Finish training.')
                model.save(current_step)  # 保存最终模型
                if tb_logger is not None:
                    tb_logger.close()  # 关闭日志记录器
                sys.exit()  # 退出程序

# 主程序入口
if __name__ == '__main__':
    main()
