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

def eval_tfp_dataset(root_spike, root_gt, save_vis=False):
    seq_metrics = {}  # 存储每个序列指标

    for seq in sorted(os.listdir(root_spike)):
        spike_dir = os.path.join(root_spike, seq, "spike")
        gt_dir    = os.path.join(root_gt, seq, "sharp")

        if not os.path.isdir(spike_dir):
            continue

        l1_sum = edge_sum = loss_sum = 0.0
        psnr_sum = ssim_sum = 0.0
        cnt = 0

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
            gt_np  = np.clip(gt_np, 0, 1)

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

        if cnt > 0:
            seq_metrics[seq] = {
                "l1": l1_sum / cnt,
                "edge": edge_sum / cnt,
                "total": loss_sum / cnt,
                "psnr": psnr_sum / cnt,
                "ssim": ssim_sum / cnt,
                "count": cnt
            }

    # -------- Find best sequence --------
    best_psnr_seq = max(seq_metrics.items(), key=lambda x: x[1]["psnr"])
    best_ssim_seq = max(seq_metrics.items(), key=lambda x: x[1]["ssim"])

    # -------- Report --------
    print("========== TFP Sequence Evaluation ==========")
    for seq, m in seq_metrics.items():
        print(f"{seq} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f} | Total Loss: {m['total']:.6f}")

    print("\n===== Best Sequences =====")
    print(f"Best PSNR sequence: {best_psnr_seq[0]} -> PSNR: {best_psnr_seq[1]['psnr']:.2f}")
    print(f"Best SSIM sequence: {best_ssim_seq[0]} -> SSIM: {best_ssim_seq[1]['ssim']:.4f}")

if __name__ == "__main__":
    eval_tfp_dataset(
        root_spike="GOPRO_Large_spike_seq/train",
        root_gt="GOPRO_Large/train",
        save_vis=False  
    )
