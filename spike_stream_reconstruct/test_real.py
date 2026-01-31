# process Gopro spike streams(middleTFP sample)
# the algorithms is in codes/utils/spike_utils.py

from codes.models.modules.arch_spkdeblur_transformer import SpkDeblurNet
from codes.utils.spike_utils import load_vidar_dat,middleISI,middleTFI,middleTFP
import torch
import cv2
import numpy as np

y = load_vidar_dat("./000226.dat", width=640,height=360)
'''
After calculation, to obtain correct spike binary images, the GoPro Spike streams obtained from the official source should have a width W of 640 and a height H that is a multiple of 360,
resulting in data with the shape [T × H × W].
'''
y = torch.tensor(y).unsqueeze(0).cuda()
print(y.shape)

y = y[:,0:88]


tfp = middleTFP(y[0].cpu().numpy(),44)
tfp = torch.tensor(tfp).unsqueeze(0).unsqueeze(0).cuda()
print(y.shape, tfp.shape)
tfp_vis = tfp[0,0].cpu().numpy() 
tfp_vis = (tfp_vis - tfp_vis.min()) / (tfp_vis.max() - tfp_vis.min())  
tfp_vis = (tfp_vis * 255).astype(np.uint8)  
cv2.imwrite("tfi_visual.png", tfp_vis)
