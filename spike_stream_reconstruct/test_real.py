from codes.models.modules.arch_spkdeblur_transformer import SpkDeblurNet
from codes.utils.spike_utils import load_vidar_dat,middleISI,middleTFI,middleTFP
import torch
import cv2
import numpy as np

y = load_vidar_dat("./000226.dat", width=640,height=360)

y = torch.tensor(y).unsqueeze(0).cuda()
print(y.shape)

y = y[:,0:88]


tfp = middleTFP(y[0].cpu().numpy(),44)
tfp = torch.tensor(tfp).unsqueeze(0).unsqueeze(0).cuda()
print(y.shape, tfp.shape)
tfp_vis = tfp[0,0].cpu().numpy()  # 取 [H, W] 的二维数组
tfp_vis = (tfp_vis - tfp_vis.min()) / (tfp_vis.max() - tfp_vis.min())  # 归一化到 0~1
tfp_vis = (tfp_vis * 255).astype(np.uint8)  # 转成 uint8
cv2.imwrite("tfi_visual.png", tfp_vis)