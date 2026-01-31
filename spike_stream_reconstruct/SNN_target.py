import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

H, W = 360, 640

# edge loss(L1+ sobel filter)
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

    pred_edge = F.conv2d(pred, sobel_x, padding=1) + \
                F.conv2d(pred, sobel_y, padding=1)
    gt_edge   = F.conv2d(gt,   sobel_x, padding=1) + \
                F.conv2d(gt,   sobel_y, padding=1)

    return F.l1_loss(pred_edge, gt_edge)


# -------------------------
# Dataset paths
# -------------------------
pred_root = "./output"              # pred_root/GOPR0XXX/xxxxxx.png
gt_root   = "GOPRO_Large/train"    # gt_root/GOPR0XXX/sharp/xxxxxx.png

seq_metrics = {} 

for seq in tqdm(sorted(os.listdir(pred_root)), desc="Sequences"):
    pred_seq_dir = os.path.join(pred_root, seq)
    gt_seq_dir   = os.path.join(gt_root, seq, "sharp")

    if not os.path.isdir(pred_seq_dir) or not os.path.isdir(gt_seq_dir):
        continue

    l1_sum = edge_sum = total_sum = 0.0
    psnr_sum = ssim_sum = 0.0
    count = 0

    for fname in sorted(os.listdir(pred_seq_dir)):
        if not fname.endswith(".png"):
            continue

        pred_path = os.path.join(pred_seq_dir, fname)
        gt_path   = os.path.join(gt_seq_dir, fname)
        if not os.path.exists(gt_path):
            continue

        # ---- Read pred ----
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.resize(pred_img, (W, H))
        pred = torch.tensor(pred_img).float() / 255.0
        pred = pred.unsqueeze(0).unsqueeze(0).to(device)

        # ---- Read GT ----
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.resize(gt_img, (W, H))
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

    if count > 0:
        seq_metrics[seq] = {
            "l1": l1_sum / count,
            "edge": edge_sum / count,
            "total": total_sum / count,
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / count,
            "count": count
        }

best_psnr_seq = max(seq_metrics.items(), key=lambda x: x[1]["psnr"])
best_ssim_seq = max(seq_metrics.items(), key=lambda x: x[1]["ssim"])

print("\n===== Sequence Evaluation =====")
for seq, m in seq_metrics.items():
    print(f"{seq} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f} | Total Loss: {m['total']:.6f}")

print("\n===== Best Sequences =====")
print(f"Best PSNR sequence: {best_psnr_seq[0]} -> PSNR: {best_psnr_seq[1]['psnr']:.2f}")
print(f"Best SSIM sequence: {best_ssim_seq[0]} -> SSIM: {best_ssim_seq[1]['ssim']:.4f}")
