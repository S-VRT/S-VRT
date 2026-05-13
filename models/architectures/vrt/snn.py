"""
SNN (Spiking Neural Network) Module for Spike Reconstruction

This module provides complete functionality for SNN-based spike reconstruction:
- Model definition (SNNResidualEnhancer)
- Dataset handling (GoProSpikeSNNDataset)
- Training functionality
- Inference/testing functionality
- Evaluation metrics (PSNR, SSIM, Loss)

Usage:
    # Training
    python -m models.architectures.vrt.snn \
        --mode train \
        --root_spike GOPRO_Large_spike_seq/train \
        --root_gt GOPRO_Large/train \
        --epochs 100 \
        --checkpoint_dir checkpoints
    
    # Testing/Inference
    python -m models.architectures.vrt.snn \
        --mode test \
        --root_spike GOPRO_Large_spike_seq/train \
        --checkpoint_path checkpoints/snn_epoch_100.pth \
        --output_dir output
    
    # Evaluation
    python -m models.architectures.vrt.snn \
        --mode eval \
        --pred_root output \
        --gt_root GOPRO_Large/train
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import snntorch as snn
from snntorch import surrogate, utils
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import project utilities
from data.spike_recc.middle_tfp.spike_utils import load_vidar_dat, middleTFP


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# GoPro spike stream dimensions
H, W = 360, 640  # Height and width for GoPro spike streams
SPIKE_WIN = 8    # SNN window size (number of time steps)


class GoProSpikeSNNDataset(Dataset):
    """Dataset for training SNN on GoPro spike sequences.
    
    Each sample contains:
        - TFP base reconstruction (from middleTFP)
        - Spike window (8 time steps around center)
        - Ground truth sharp image
    """
    
    def __init__(self, root_spike, root_gt):
        """
        Args:
            root_spike (str): Root directory containing spike sequences
                Structure: root_spike/{seq}/spike/*.dat
            root_gt (str): Root directory containing ground truth images
                Structure: root_gt/{seq}/sharp/*.png
        """
        self.samples = []
        for seq in sorted(os.listdir(root_spike)):
            spike_dir = os.path.join(root_spike, seq, "spike")
            gt_dir = os.path.join(root_gt, seq, "sharp")
            if not os.path.isdir(spike_dir) or not os.path.isdir(gt_dir):
                continue
            for f in sorted(os.listdir(spike_dir)):
                if f.endswith(".dat"):
                    sp = os.path.join(spike_dir, f)
                    gt = os.path.join(gt_dir, f.replace(".dat", ".png"))
                    if os.path.exists(gt):
                        self.samples.append((sp, gt))
        print(f"[GoProSpikeSNNDataset] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _extract_spike_window(self, spike, center, win=SPIKE_WIN):
        """Extract a window of spike frames around the center time step.
        
        Args:
            spike: Spike sequence [T, H, W]
            center: Center time step index
            win: Window size (default: SPIKE_WIN=8)
            
        Returns:
            Spike window [win, H, W], padded if necessary
        """
        half = win // 2
        l = max(center - half, 0)
        r = min(center + half, spike.shape[0])
        out = spike[l:r]
        if out.shape[0] < win:
            out = np.pad(out, ((0, win - out.shape[0]), (0, 0), (0, 0)))
        return out

    def __getitem__(self, idx):
        """Get a training sample.
        
        Returns:
            tuple: (tfp, spike_window, gt)
                - tfp: TFP base reconstruction [1, H, W], normalized [0, 1]
                - spike_window: Spike window [win, H, W], float32
                - gt: Ground truth image [1, H, W], normalized [0, 1]
        """
        spike_path, gt_path = self.samples[idx]
        
        # Load spike data
        spike = load_vidar_dat(spike_path, width=W, height=H)
        assert spike.shape[1:] == (H, W), f"Spike size error: {spike.shape}"
        
        # Extract center time step and spike window
        mid = spike.shape[0] // 2
        spike_window = self._extract_spike_window(spike, mid)  # [win, H, W]
        spike_window = torch.from_numpy(spike_window).float()
        
        # Compute TFP base reconstruction
        tfp = middleTFP(spike, mid)
        tfp = (tfp - tfp.min()) / (tfp.max() - tfp.min() + 1e-6)
        tfp = torch.from_numpy(tfp).float().unsqueeze(0)
        
        # Load ground truth
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.shape != (H, W):
            gt = cv2.resize(gt, (W, H))
        gt = torch.from_numpy(gt).float().unsqueeze(0) / 255.0
        
        return tfp, spike_window, gt


def sobel_l1_loss(pred, target):
    """Compute L1 loss on Sobel-filtered edges.
    
    This loss encourages the model to preserve edge details.
    
    Args:
        pred: Predicted image [B, 1, H, W]
        target: Target image [B, 1, H, W]
        
    Returns:
        L1 loss between Sobel-filtered predictions and targets
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32, device=pred.device
    ).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]],
        dtype=torch.float32, device=pred.device
    ).view(1, 1, 3, 3)
    
    pred_edge = F.conv2d(pred, sobel_x, padding=1) + F.conv2d(pred, sobel_y, padding=1)
    target_edge = F.conv2d(target, sobel_x, padding=1) + F.conv2d(target, sobel_y, padding=1)
    
    return F.l1_loss(pred_edge, target_edge)


def edge_loss(pred, gt):
    """Edge loss (L1 + Sobel filter) - alternative implementation.
    
    Args:
        pred: Predicted image [B, 1, H, W] or [1, 1, H, W]
        gt: Ground truth image [B, 1, H, W] or [1, 1, H, W]
        
    Returns:
        L1 loss between Sobel-filtered edges
    """
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=pred.dtype, device=pred.device
    ).view(1, 1, 3, 3)

    pred_edge = F.conv2d(pred, sobel_x, padding=1) + F.conv2d(pred, sobel_y, padding=1)
    gt_edge = F.conv2d(gt, sobel_x, padding=1) + F.conv2d(gt, sobel_y, padding=1)

    return F.l1_loss(pred_edge, gt_edge)


class SNNResidualEnhancer(nn.Module):
    """Spiking Neural Network for residual enhancement of TFP reconstruction.
    
    The network processes a sequence of spike frames and outputs a residual image
    that is added to the TFP base reconstruction to improve quality.
    """
    
    def __init__(self, beta=0.9):
        """
        Args:
            beta (float): Leaky integrate-and-fire neuron decay parameter.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, spike_seq):
        """Forward pass through the SNN.
        
        Args:
            spike_seq: Spike sequence [B, T, H, W]
            
        Returns:
            Residual image [B, 1, H, W]
        """
        utils.reset(self)
        mem1 = mem2 = None
        for t in range(spike_seq.shape[1]):
            x = spike_seq[:, t:t+1]  # [B, 1, H, W]
            x = self.conv1(x)
            x, mem1 = self.lif1(x, mem1)
            x = self.conv2(x)
            x, mem2 = self.lif2(x, mem2)
        return self.conv3(x)


def train(root_spike, root_gt, epochs=100, batch_size=1, lr=2e-4, 
          checkpoint_dir="checkpoints", device=None, seed=42):
    """Train the SNN Residual Enhancer model.
    
    Args:
        root_spike (str): Root directory for spike sequences
        root_gt (str): Root directory for ground truth images
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        checkpoint_dir (str): Directory to save model checkpoints
        device (str): Device to use ('cuda' or 'cpu'). If None, auto-detect.
        seed (int): Random seed for reproducibility
        
    Returns:
        Trained model
    """
    set_seed(seed)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[Training] Using device: {device}")
    
    # Create dataset and dataloader
    dataset = GoProSpikeSNNDataset(root_spike, root_gt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = SNNResidualEnhancer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[Training] Checkpoints will be saved to: {checkpoint_dir}")
    
    # Training loop
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0
        num_batches = 0
        
        for tfp, spk, gt in loader:
            tfp, spk, gt = tfp.to(device), spk.to(device), gt.to(device)
            
            # Forward pass
            res = model(spk)  # Predict residual
            out = torch.clamp(tfp + res, 0, 1)  # Combine TFP + residual
            
            # Compute loss
            loss = F.l1_loss(out, gt) + 0.1 * sobel_l1_loss(out, gt)
            
            # Backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_sum += loss.item()
            num_batches += 1
        
        avg_loss = loss_sum / num_batches if num_batches > 0 else 0.0
        print(f"[Epoch {ep:03d}/{epochs}] Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"snn_epoch_{ep}.pth")
        torch.save(model.state_dict(), checkpoint_path)
    
    print(f"[Training] Training completed. Final checkpoint: {checkpoint_dir}/snn_epoch_{epochs}.pth")
    return model


@torch.no_grad()
def test(root_spike_seq, checkpoint_path, out_root=None, save_image=False, 
         return_result=False, device=None):
    """Test/inference with trained SNN model.
    
    Args:
        root_spike_seq (str): Root directory containing spike sequences
        checkpoint_path (str): Path to trained model checkpoint
        out_root (str): Output directory for saving images. If None, images won't be saved.
        save_image (bool): Whether to save reconstructed images
        return_result (bool): Whether to return results as a list
        device (str): Device to use ('cuda' or 'cpu'). If None, auto-detect.
        
    Returns:
        If return_result=True, returns list of dictionaries with reconstruction results.
        Otherwise, returns None.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[Testing] Using device: {device}")
    
    # Load model
    model = SNNResidualEnhancer().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"[Testing] Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    results_list = []
    
    if save_image and out_root is not None:
        os.makedirs(out_root, exist_ok=True)
    
    for seq in sorted(os.listdir(root_spike_seq)):
        spike_dir = os.path.join(root_spike_seq, seq, "spike")
        if not os.path.isdir(spike_dir):
            continue
        
        out_seq_dir = os.path.join(out_root, seq) if out_root else None
        if save_image and out_seq_dir:
            os.makedirs(out_seq_dir, exist_ok=True)
        
        for fname in sorted(os.listdir(spike_dir)):
            if not fname.endswith(".dat"):
                continue
            
            dat_path = os.path.join(spike_dir, fname)
            
            # Load spike data
            spike = load_vidar_dat(dat_path, width=W, height=H)
            center = spike.shape[0] // 2
            
            # Extract spike window
            spike_window = spike[
                center - SPIKE_WIN // 2 : center + SPIKE_WIN // 2
            ]
            if spike_window.shape[0] < SPIKE_WIN:
                spike_window = np.pad(
                    spike_window,
                    ((0, SPIKE_WIN - spike_window.shape[0]), (0, 0), (0, 0))
                )
            spike_window = torch.from_numpy(spike_window).float().unsqueeze(0).to(device)
            
            # Compute TFP base
            tfp = middleTFP(spike, center)
            tfp = (tfp - tfp.min()) / (tfp.max() - tfp.min() + 1e-6)
            tfp = torch.from_numpy(tfp).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # SNN inference
            res = model(spike_window)
            out = torch.clamp(tfp + res, 0, 1)
            
            # Save image if requested
            if save_image and out_seq_dir:
                out_img = (out[0, 0].cpu().numpy() * 255).astype(np.uint8)
                out_path = os.path.join(out_seq_dir, fname.replace(".dat", ".png"))
                cv2.imwrite(out_path, out_img)
            
            # Collect results if requested
            if return_result:
                results_list.append({
                    "seq": seq,
                    "name": fname.replace(".dat", ""),
                    "pred": out.detach()  # [1, 1, H, W]
                })
    
    if not return_result:
        print("[Testing] Finished")
    
    if return_result:
        return results_list


def evaluate(pred_root, gt_root, device=None):
    """Evaluate SNN reconstruction results.
    
    Computes PSNR, SSIM, and loss metrics for each sequence.
    
    Args:
        pred_root (str): Root directory containing predicted images
            Structure: pred_root/{seq}/*.png
        gt_root (str): Root directory containing ground truth images
            Structure: gt_root/{seq}/sharp/*.png
        device (str): Device to use ('cuda' or 'cpu'). If None, auto-detect.
        
    Returns:
        Dictionary mapping sequence names to metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[Evaluation] Using device: {device}")
    
    seq_metrics = {}
    
    for seq in tqdm(sorted(os.listdir(pred_root)), desc="Evaluating sequences"):
        pred_seq_dir = os.path.join(pred_root, seq)
        gt_seq_dir = os.path.join(gt_root, seq, "sharp")
        
        if not os.path.isdir(pred_seq_dir) or not os.path.isdir(gt_seq_dir):
            continue
        
        l1_sum = edge_sum = total_sum = 0.0
        psnr_sum = ssim_sum = 0.0
        count = 0
        
        for fname in sorted(os.listdir(pred_seq_dir)):
            if not fname.endswith(".png"):
                continue
            
            pred_path = os.path.join(pred_seq_dir, fname)
            gt_path = os.path.join(gt_seq_dir, fname)
            if not os.path.exists(gt_path):
                continue
            
            # Read predicted image
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            pred_img = cv2.resize(pred_img, (W, H))
            pred = torch.tensor(pred_img).float() / 255.0
            pred = pred.unsqueeze(0).unsqueeze(0).to(device)
            
            # Read GT image
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt_img = cv2.resize(gt_img, (W, H))
            gt = torch.tensor(gt_img).float() / 255.0
            gt = gt.unsqueeze(0).unsqueeze(0).to(device)
            
            # Compute losses
            l1 = F.l1_loss(pred, gt)
            edge = edge_loss(pred, gt)
            loss = l1 + 0.1 * edge
            
            # Compute PSNR / SSIM
            pred_np = pred[0, 0].cpu().numpy()
            gt_np = gt[0, 0].cpu().numpy()
            psnr_val = psnr(gt_np, pred_np, data_range=1.0)
            ssim_val = ssim(gt_np, pred_np, data_range=1.0)
            
            # Accumulate
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
    
    # Print results
    print("\n===== Sequence Evaluation =====")
    for seq, m in seq_metrics.items():
        print(f"{seq} | PSNR: {m['psnr']:.2f} | SSIM: {m['ssim']:.4f} | Total Loss: {m['total']:.6f}")
    
    if seq_metrics:
        best_psnr_seq = max(seq_metrics.items(), key=lambda x: x[1]["psnr"])
        best_ssim_seq = max(seq_metrics.items(), key=lambda x: x[1]["ssim"])
        
        print("\n===== Best Sequences =====")
        print(f"Best PSNR sequence: {best_psnr_seq[0]} -> PSNR: {best_psnr_seq[1]['psnr']:.2f}")
        print(f"Best SSIM sequence: {best_ssim_seq[0]} -> SSIM: {best_ssim_seq[1]['ssim']:.4f}")
    
    return seq_metrics


def main():
    """Main entry point for SNN script."""
    parser = argparse.ArgumentParser(description="SNN for Spike Reconstruction")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "eval"],
                       help="Mode: 'train', 'test', or 'eval'")
    
    # Training arguments
    parser.add_argument("--root_spike", type=str, default=None,
                       help="Root directory for spike sequences")
    parser.add_argument("--root_gt", type=str, default=None,
                       help="Root directory for ground truth images")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for training (default: 1)")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate (default: 2e-4)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints (default: checkpoints)")
    
    # Testing arguments
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for saving reconstructed images")
    parser.add_argument("--save_image", action="store_true",
                       help="Save reconstructed images during testing")
    
    # Evaluation arguments
    parser.add_argument("--pred_root", type=str, default=None,
                       help="Root directory containing predicted images")
    parser.add_argument("--gt_root", type=str, default=None,
                       help="Root directory containing ground truth images")
    
    # Common arguments
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use ('cuda' or 'cpu'). Auto-detect if not specified.")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.root_spike is None or args.root_gt is None:
            parser.error("--root_spike and --root_gt are required for training mode")
        train(
            root_spike=args.root_spike,
            root_gt=args.root_gt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            seed=args.seed
        )
    
    elif args.mode == "test":
        if args.root_spike is None or args.checkpoint_path is None:
            parser.error("--root_spike and --checkpoint_path are required for test mode")
        test(
            root_spike_seq=args.root_spike,
            checkpoint_path=args.checkpoint_path,
            out_root=args.output_dir,
            save_image=args.save_image,
            return_result=False,
            device=args.device
        )
    
    elif args.mode == "eval":
        if args.pred_root is None or args.gt_root is None:
            parser.error("--pred_root and --gt_root are required for eval mode")
        evaluate(
            pred_root=args.pred_root,
            gt_root=args.gt_root,
            device=args.device
        )


if __name__ == "__main__":
    main()
