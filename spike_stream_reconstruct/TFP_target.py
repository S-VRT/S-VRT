import torch
import torch.nn.functional as F
import cv2
import numpy as np

from codes.utils.spike_utils import load_vidar_dat, middleTFP
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def edge_loss(pred, gt):
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    pred_edge = F.conv2d(pred, sobel_x, padding=1) + F.conv2d(pred, sobel_y, padding=1)
    gt_edge   = F.conv2d(gt,   sobel_x, padding=1) + F.conv2d(gt,   sobel_y, padding=1)

    return F.l1_loss(pred_edge, gt_edge)


spike = load_vidar_dat(
    "./000226.dat",
    width=640,
    height=360
)  # [T, H, W]

spike = torch.tensor(spike).cuda()
spike = spike[:88]  


tfp = middleTFP(
    spike.cpu().numpy(),
    44
)  # [H, W]

tfp = torch.tensor(tfp).unsqueeze(0).unsqueeze(0).cuda()
# shape: [1, 1, 360, 640]



gt = cv2.imread("./000226.png", cv2.IMREAD_GRAYSCALE)
gt = cv2.resize(gt, (640, 360))
gt = torch.tensor(gt).float() / 255.0
gt = gt.unsqueeze(0).unsqueeze(0).cuda()


l1 = F.l1_loss(tfp, gt)
edge = edge_loss(tfp, gt)
loss_tfp = l1 + 0.1 * edge

print("TFP L1 Loss   :", l1.item())
print("TFP Edge Loss :", edge.item())
print("TFP Total Loss:", loss_tfp.item())


tfp_vis = tfp[0, 0].detach().cpu().numpy()
tfp_vis = (tfp_vis - tfp_vis.min()) / (tfp_vis.max() - tfp_vis.min() + 1e-6)
tfp_vis = (tfp_vis * 255).astype(np.uint8)

cv2.imwrite("tfp_visual.png", tfp_vis)

tfp_np = tfp[0, 0].detach().cpu().numpy()
gt_np  = gt[0, 0].detach().cpu().numpy()

tfp_np = np.clip(tfp_np, 0, 1)
gt_np  = np.clip(gt_np,  0, 1)

psnr_tfp = psnr(gt_np, tfp_np, data_range=1.0)
ssim_tfp = ssim(gt_np, tfp_np, data_range=1.0)

print(f"TFP PSNR: {psnr_tfp:.2f} dB")
print(f"TFP SSIM: {ssim_tfp:.4f}")