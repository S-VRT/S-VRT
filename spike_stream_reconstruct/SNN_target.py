import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Edge Loss
# -------------------------
def edge_loss(pred, gt):
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    pred_edge = F.conv2d(pred, sobel_x, padding=1) + F.conv2d(pred, sobel_y, padding=1)
    gt_edge   = F.conv2d(gt,   sobel_x, padding=1) + F.conv2d(gt,   sobel_y, padding=1)

    return F.l1_loss(pred_edge, gt_edge)

# -------------------------
# 1. 读取算法输出
# -------------------------
pred_img = cv2.imread("snn_enhanced.png", cv2.IMREAD_GRAYSCALE)
assert pred_img is not None, "Result image not found"

pred_img = cv2.resize(pred_img, (640, 360))
pred = torch.tensor(pred_img).float() / 255.0
pred = pred.unsqueeze(0).unsqueeze(0).to(device)

# -------------------------
# 2. 读取 GT
# -------------------------
gt_img = cv2.imread("./000226.png", cv2.IMREAD_GRAYSCALE)
assert gt_img is not None, "GT image not found"

gt_img = cv2.resize(gt_img, (640, 360))
gt = torch.tensor(gt_img).float() / 255.0
gt = gt.unsqueeze(0).unsqueeze(0).to(device)

# -------------------------
# 3. Loss
# -------------------------
l1 = F.l1_loss(pred, gt)
edge = edge_loss(pred, gt)
loss = l1 + 0.1 * edge

# -------------------------
# 4. PSNR / SSIM
# -------------------------
pred_np = pred[0, 0].cpu().numpy()
gt_np   = gt[0, 0].cpu().numpy()

psnr_val = psnr(gt_np, pred_np, data_range=1.0)
ssim_val = ssim(gt_np, pred_np, data_range=1.0)

# -------------------------
# 5. Print
# -------------------------
print("L1 Loss   :", l1.item())
print("Edge Loss :", edge.item())
print("Total Loss:", loss.item())
print(f"PSNR      : {psnr_val:.2f} dB")
print(f"SSIM      : {ssim_val:.4f}")
