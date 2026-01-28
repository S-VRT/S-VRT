import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import snntorch as snn
from snntorch import surrogate, utils
from codes.utils.spike_utils import load_vidar_dat, middleTFP

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

H, W = 360, 640
SPIKE_WIN = 8  # SNN windows

class GoProSpikeSNNDataset(Dataset):
    def __init__(self, root_spike, root_gt):
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
        print(f"[Dataset] total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _extract_spike_window(self, spike, center, win=SPIKE_WIN):
        half = win // 2
        l = max(center - half, 0)
        r = min(center + half, spike.shape[0])
        out = spike[l:r]
        if out.shape[0] < win:
            out = np.pad(out, ((0, win - out.shape[0]), (0,0), (0,0)))
        return out

    def __getitem__(self, idx):
        spike_path, gt_path = self.samples[idx]
        spike = load_vidar_dat(spike_path, width=W, height=H)
        assert spike.shape[1:] == (H, W), f"Spike size error: {spike.shape}"

        mid = spike.shape[0] // 2
        spike_window = self._extract_spike_window(spike, mid)  # [win,H,W]
        spike_window = torch.from_numpy(spike_window).float()

        tfp = middleTFP(spike, mid)
        tfp = (tfp - tfp.min()) / (tfp.max() - tfp.min() + 1e-6)
        tfp = torch.from_numpy(tfp).float().unsqueeze(0)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.shape != (H, W):
            gt = cv2.resize(gt, (W, H))
        gt = torch.from_numpy(gt).float().unsqueeze(0)/255.0

        return tfp, spike_window, gt


def sobel_l1_loss(pred, target):
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    return F.l1_loss(
        F.conv2d(pred, sobel_x, padding=1) + F.conv2d(pred, sobel_y, padding=1),
        F.conv2d(target, sobel_x, padding=1) + F.conv2d(target, sobel_y, padding=1)
    )


# ===============================
# 3️⃣ SNN Residual Enhancer
# ===============================
class SNNResidualEnhancer(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, spike_seq):
        """
        spike_seq: [B, T, H, W]
        """
        utils.reset(self)
        mem1 = mem2 = None
        for t in range(spike_seq.shape[1]):
            x = spike_seq[:, t:t+1]  # [B,1,H,W]
            x = self.conv1(x)
            x, mem1 = self.lif1(x, mem1)
            x = self.conv2(x)
            x, mem2 = self.lif2(x, mem2)
        return self.conv3(x)

# ===============================
# 4️⃣ Training
# ===============================
def train(root_spike, root_gt, epochs=20, batch_size=1, lr=2e-4):
    dataset = GoProSpikeSNNDataset(root_spike, root_gt)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SNNResidualEnhancer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        loss_sum = 0
        for tfp, spk, gt in loader:
            tfp, spk, gt = tfp.to(device), spk.to(device), gt.to(device)

            res = model(spk)
            out = torch.clamp(tfp + res, 0, 1)

            loss = F.l1_loss(out, gt) + 0.1 * sobel_l1_loss(out, gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        print(f"[Epoch {ep:03d}] Loss {loss_sum/len(loader):.6f}")
        torch.save(model.state_dict(), f"checkpoints/snn_epoch_{ep}.pth")

# ===============================
# 5️⃣ Inference
# ===============================
@torch.no_grad()
def infer(dat_path, out_path):
    spike = load_vidar_dat(dat_path, width=W, height=H)
    center = spike.shape[0] // 2

    spike_window = spike[center - SPIKE_WIN//2 : center + SPIKE_WIN//2]
    spike_window = torch.from_numpy(spike_window).float().unsqueeze(0).to(device)

    tfp = middleTFP(spike, center)
    tfp = (tfp - tfp.min()) / (tfp.max() - tfp.min() + 1e-6)
    tfp = torch.from_numpy(tfp).float().unsqueeze(0).unsqueeze(0).to(device)

    model = SNNResidualEnhancer().to(device)
    model.load_state_dict(torch.load("checkpoints/snn_epoch_20.pth", map_location=device))
    model.eval()

    out = torch.clamp(tfp + model(spike_window), 0, 1)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite(out_path, (out[0,0].cpu().numpy()*255).astype(np.uint8))
    print(f"✔ Saved {out_path}")

# ===============================
# 6️⃣ Main
# ===============================
if __name__ == "__main__":
    train(
        root_spike="GOPRO_Large_spike_seq/train",
        root_gt="GOPRO_Large/train",
        epochs=20
    )

    infer(
        "GOPRO_Large_spike_seq/train/GOPR0884_11_00/spike/000226.dat",
        "output/snn_enhanced.png"
    )
