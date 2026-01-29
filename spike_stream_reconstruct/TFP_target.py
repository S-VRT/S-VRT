import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np

from codes.utils.spike_utils import load_vidar_dat, middleTFP
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

H, W = 360, 640
SPIKE_LEN = 88
CENTER = 44


# -------------------------
# Edge Loss
# -------------------------
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

    pred_edge = F.conv2d(pred, sobel_x, padding=1) + \
                F.conv2d(pred, sobel_y, padding=1)
    gt_edge   = F.conv2d(gt,   sobel_x, padding=1) + \
                F.conv2d(gt,   sobel_y, padding=1)

    return F.l1_loss(pred_edge, gt_edge)


# -------------------------
# Dataset-level evaluation
# -------------------------
def eval_tfp_dataset(root_spike, root_gt, save_vis=False):
    l1_sum = edge_sum = loss_sum = 0.0
    psnr_sum = ssim_sum = 0.0
    cnt = 0

    for seq in sorted(os.listdir(root_spike)):
        spike_dir = os.path.join(root_spike, seq, "spike")
        gt_dir    = os.path.join(root_gt, seq, "sharp")

        if not os.path.isdir(spike_dir):
            continue

        for fname in sorted(os.listdir(spike_dir)):
            if not fname.endswith(".dat"):
                continue

            dat_path = os.path.join(spike_dir, fname)
            gt_path  = os.path.join(gt_dir, fname.replace(".dat", ".png"))

            if not os.path.exists(gt_path):
                continue

            # -------- spike --------
            spike = load_vidar_dat(dat_path, width=W, height=H)
            spike = torch.tensor(spike[:SPIKE_LEN]).to(device)

            # -------- TFP --------
            tfp = middleTFP(spike.cpu().numpy(), CENTER)
            tfp = torch.tensor(tfp).float().to(device)
            tfp = tfp.unsqueeze(0).unsqueeze(0)

            # -------- GT --------
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (W, H))
            gt = torch.tensor(gt).float().to(device) / 255.0
            gt = gt.unsqueeze(0).unsqueeze(0)

            # -------- Loss --------
            l1 = F.l1_loss(tfp, gt)
            edge = edge_loss(tfp, gt)
            loss = l1 + 0.1 * edge

            # -------- Metrics --------
            tfp_np = tfp[0, 0].detach().cpu().numpy()
            gt_np  = gt[0, 0].detach().cpu().numpy()

            tfp_np = np.clip(tfp_np, 0, 1)
            gt_np  = np.clip(gt_np,  0, 1)

            psnr_val = psnr(gt_np, tfp_np, data_range=1.0)
            ssim_val = ssim(gt_np, tfp_np, data_range=1.0)

            # -------- Accumulate --------
            l1_sum += l1.item()
            edge_sum += edge.item()
            loss_sum += loss.item()
            psnr_sum += psnr_val
            ssim_sum += ssim_val
            cnt += 1

            # -------- Optional save --------
            if save_vis:
                vis = (tfp_np - tfp_np.min()) / (tfp_np.max() - tfp_np.min() + 1e-6)
                vis = (vis * 255).astype(np.uint8)
                os.makedirs("tfp_vis", exist_ok=True)
                cv2.imwrite(f"tfp_vis/{seq}_{fname[:-4]}.png", vis)

    # -------- Final report --------
    print("========== TFP Dataset Evaluation ==========")
    print(f"Samples     : {cnt}")
    print(f"L1 Loss     : {l1_sum / cnt:.6f}")
    print(f"Edge Loss   : {edge_sum / cnt:.6f}")
    print(f"Total Loss  : {loss_sum / cnt:.6f}")
    print(f"PSNR        : {psnr_sum / cnt:.2f} dB")
    print(f"SSIM        : {ssim_sum / cnt:.4f}")
    print("============================================")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    eval_tfp_dataset(
        root_spike="GOPRO_Large_spike_seq/train",
        root_gt="GOPRO_Large/train",
        save_vis=False   # True 就会保存 TFP 可视化
    )
