import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

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
# Dataset paths
# -------------------------
pred_dir = "./pred_dir"   # 预测结果文件夹
gt_dir   = "./gt_dir"     # GT 文件夹

pred_files = sorted(os.listdir(pred_dir))

# -------------------------
# Metrics accumulator
# -------------------------
l1_sum = 0.0
edge_sum = 0.0
total_sum = 0.0
psnr_sum = 0.0
ssim_sum = 0.0
count = 0

# -------------------------
# Loop over dataset
# -------------------------
for fname in tqdm(pred_files):
    pred_path = os.path.join(pred_dir, fname)
    gt_path   = os.path.join(gt_dir, fname)

    if not os.path.exists(gt_path):
        print(f"[Skip] GT not found for {fname}")
        continue

    # ---- Read pred ----
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    if pred_img is None:
        continue
    pred_img = cv2.resize(pred_img, (640, 360))
    pred = torch.tensor(pred_img).float() / 255.0
    pred = pred.unsqueeze(0).unsqueeze(0).to(device)

    # ---- Read GT ----
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt_img is None:
        continue
    gt_img = cv2.resize(gt_img, (640, 360))
    gt = torch.tensor(gt_img).float() / 255.0
    gt = gt.unsqueeze(0).unsqueeze(0).to(device)

    # ---- Loss ----
    l1 = F.l1_loss(pred, gt)
    edge = edge_loss(pred, gt)
    loss = l1 + 0.1 * edge

    # ---- PSNR / SSIM ----
    pred_np = pred[0, 0].cpu().numpy()
    gt_np   = gt[0, 0].cpu().numpy()

    psnr_val = psnr(gt_np, pred_np, data_range=1.0)
    ssim_val = ssim(gt_np, pred_np, data_range=1.0)

    # ---- Accumulate ----
    l1_sum += l1.item()
    edge_sum += edge.item()
    total_sum += loss.item()
    psnr_sum += psnr_val
    ssim_sum += ssim_val
    count += 1

# -------------------------
# Final results
# -------------------------
print("\n===== Dataset Evaluation =====")
print(f"Samples    : {count}")
print(f"L1 Loss    : {l1_sum / count:.6f}")
print(f"Edge Loss  : {edge_sum / count:.6f}")
print(f"Total Loss : {total_sum / count:.6f}")
print(f"PSNR       : {psnr_sum / count:.2f} dB")
print(f"SSIM       : {ssim_sum / count:.4f}")

