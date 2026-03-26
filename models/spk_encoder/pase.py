import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelPredictor(nn.Module):
    """Kernel Predictor

    Given a spike stream S0 [B, L, H, W], predict per‑pixel convolution
    kernels for C branches:
        K: kernel_size
        L: temporal length
        C: out_channels (number of branches)

    Output shape: [B, C * (L*K*K), H, W]
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        hidden_chans: int = 32,
    ) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.hidden_chans = hidden_chans

        # A simple conv stack; you can later replace this with a deeper CNN if needed.
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_chans, hidden_chans, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Predict C * (L*K*K) kernel weights per pixel
            nn.Conv2d(
                hidden_chans,
                out_chans * (in_chans * kernel_size * kernel_size),
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: spike stream S0, shape [B, L, H, W]

        Returns:
            kernel_logits: [B, C * (L*K*K), H, W]
        """
        assert x.dim() == 4, f"Expected [B, L, H, W], got {x.shape}"
        return self.net(x)


class PixelAdaptiveSpikeEncoder(nn.Module):
    """Pixel-Adaptive Spike Encoder (PASE)

    对应 `spk_encoder_mermaid.txt` 中的结构：
      - 输入：Spike Stream S0, 形状 [B, L, H, W]
      - Z 区域：在像素 (x, y) 上的 Pixel-Adaptive Convolution
          * Spike Patch K×K×L
          * Predicted Kernel K×K×L
          * Dot Product -> y(x, y)
      - PASE 主体：C 个并行分支
          * 每个分支：Kernel Predictor -> Conv Kernel Tensor -> Conv -> Feature Map
      - 输出：特征图堆叠 Y, 形状 [B, C, H, W]
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        kernel_size: int = 3,
        hidden_chans: int = 32,
        normalize_kernel: bool = True,
    ) -> None:
        """
        Args:
            in_chans: L, spike 时间维长度
            out_chans: C, 输出特征通道数（对应 C 个分支）
            kernel_size: K, 空间卷积核尺寸
            hidden_chans: Kernel Predictor 内部通道数
            normalize_kernel: 是否对每个像素的 K×K×L kernel 做 softmax 归一化
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.normalize_kernel = normalize_kernel

        self.kernel_predictor = KernelPredictor(
            in_chans=in_chans,
            out_chans=out_chans,
            kernel_size=kernel_size,
            hidden_chans=hidden_chans,
        )

    def forward(self, spike_stream: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_stream: S0, spike stream，形状 [B, L, H, W]

        Returns:
            features: 堆叠后的特征图，形状 [B, C, H, W]
        """
        assert spike_stream.dim() == 4, f"Expected [B, L, H, W], got {spike_stream.shape}"
        b, l, h, w = spike_stream.shape
        k = self.kernel_size
        c = self.out_chans

        # ============================
        # 1) 提取每个像素对应的 K×K×L patch
        #    使用 unfold 生成 [B, L*K*K, H*W]
        # ============================
        patches = F.unfold(
            spike_stream,
            kernel_size=k,
            padding=k // 2,
        )  # [B, L*K*K, H*W]

        # ============================
        # 2) Kernel Predictor：产生 per-pixel kernel
        #    kernel_logits: [B, C*(L*K*K), H, W]
        # ============================
        kernel_logits = self.kernel_predictor(spike_stream)  # [B, C*(L*K*K), H, W]

        # 展开成 [B, C, L*K*K, H*W]
        kernel_logits = kernel_logits.view(
            b, c, l * k * k, h * w
        )  # [B, C, L*K*K, H*W]

        # ============================
        # 3) 可选：对每个像素的 kernel 做 softmax 归一化
        # ============================
        if self.normalize_kernel:
            kernel = F.softmax(kernel_logits, dim=2)
        else:
            kernel = kernel_logits

        # ============================
        # 4) Dot Product：K×K×L patch ⊗ K×K×L kernel
        #    patches: [B,      L*K*K, H*W] -> [B, 1, L*K*K, H*W]
        #    kernel:  [B, C,   L*K*K, H*W]
        #    out:     [B, C,          H*W]
        # ============================
        patches = patches.unsqueeze(1)  # [B, 1, L*K*K, H*W]
        out = (kernel * patches).sum(dim=2)  # [B, C, H*W]

        # ============================
        # 5) reshape 回 feature map: [B, C, H, W]
        # ============================
        features = out.view(b, c, h, w)
        return features


__all__ = ["PixelAdaptiveSpikeEncoder", "KernelPredictor"]

