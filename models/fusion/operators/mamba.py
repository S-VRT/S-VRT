from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from contextlib import contextmanager, nullcontext


class _MambaBlock(nn.Module):
    def __init__(self, model_dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except (ImportError, ModuleNotFoundError):
            self.mamba = None
            return
        self.norm = nn.LayerNorm(model_dim)
        self.mamba = Mamba(d_model=model_dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.mamba is None:
            raise RuntimeError("mamba_ssm is required for mamba fusion operator.")
        if not tokens.is_cuda:
            raise RuntimeError("mamba_ssm is required for mamba fusion operator with CUDA tensors.")
        return tokens + self.mamba(self.norm(tokens))


class MambaFusionOperator(nn.Module):
    expects_structured_early = True
    frame_contract = "collapsed"

    def __init__(self, rgb_chans: int, spike_chans: int, out_chans: int, operator_params: Dict):
        super().__init__()
        if rgb_chans != 3:
            raise ValueError("MambaFusionOperator requires rgb_chans=3.")
        if spike_chans != 1:
            raise ValueError("MambaFusionOperator requires spike_chans=1 at construction time.")
        if out_chans != 3:
            raise ValueError("MambaFusionOperator requires out_chans=3.")

        token_dim = int(operator_params.get("token_dim", operator_params.get("model_dim", 48)))
        token_stride = int(operator_params.get("token_stride", 4))
        d_state = int(operator_params.get("d_state", 32))
        d_conv = int(operator_params.get("d_conv", 4))
        expand = int(operator_params.get("expand", 2))
        num_layers = int(operator_params.get("num_layers", 3))
        token_chunk_size = int(operator_params.get("token_chunk_size", 0))
        alpha_init = float(operator_params.get("alpha_init", 0.05))
        gate_bias_init = float(operator_params.get("gate_bias_init", operator_params.get("init_gate_bias", -2.0)))
        enable_diagnostics = bool(operator_params.get("enable_diagnostics", False))
        mamba_amp_policy = str(operator_params.get("mamba_amp_policy", "fp32")).strip().lower()
        if mamba_amp_policy not in {"fp32", "autocast", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mamba_amp_policy={mamba_amp_policy!r}; "
                "expected one of: fp32, autocast, fp16, bf16"
            )

        self.enable_diagnostics = enable_diagnostics
        self.token_chunk_size = token_chunk_size
        self.mamba_amp_policy = mamba_amp_policy
        self.rgb_context_encoder = nn.Sequential(
            nn.Conv2d(3, token_dim, kernel_size=3, stride=token_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spike_token_encoder = nn.Sequential(
            nn.Conv2d(1, token_dim, kernel_size=3, stride=token_stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
        )
        self.mamba_token_mixer = nn.ModuleList(
            [_MambaBlock(model_dim=token_dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)]
        )
        self.fusion_writeback_head = nn.ModuleDict(
            {
                "body": nn.Sequential(
                    nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                "delta": nn.Conv2d(token_dim, 3, kernel_size=1),
                "gate": nn.Conv2d(token_dim, 3, kernel_size=1),
            }
        )
        self.alpha = nn.Parameter(torch.full((1, 3, 1, 1), alpha_init))
        self._warmup_stage = "full"
        self._last_diagnostics: dict = {"warmup_stage": "full"}

        nn.init.normal_(self.fusion_writeback_head["delta"].weight, std=1e-3)
        nn.init.zeros_(self.fusion_writeback_head["delta"].bias)
        nn.init.normal_(self.fusion_writeback_head["gate"].weight, std=1e-3)
        nn.init.constant_(self.fusion_writeback_head["gate"].bias, gate_bias_init)
        self.timer = None

    def set_timer(self, timer) -> None:
        self.timer = timer

    def _timer(self, name: str):
        timer = getattr(self, "timer", None)
        if timer is None:
            return nullcontext()
        return timer.timer(name)

    @contextmanager
    def _profiled_timer(self, name: str):
        with torch.profiler.record_function(name):
            with self._timer(name):
                yield

    def set_warmup_stage(self, stage) -> None:
        normalized = "full" if stage in {None, "", "full"} else str(stage).strip().lower()
        if normalized not in {"full", "writeback_only", "token_mixer"}:
            raise ValueError(f"Unsupported Mamba warmup stage: {stage!r}")
        self._warmup_stage = normalized
        token_trainable = normalized != "writeback_only"
        for module in (self.rgb_context_encoder, self.spike_token_encoder, self.mamba_token_mixer):
            for param in module.parameters():
                param.requires_grad_(token_trainable)
        for param in self.fusion_writeback_head.parameters():
            param.requires_grad_(True)
        self.alpha.requires_grad_(True)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    @staticmethod
    def _device_autocast_disabled_context(tensor: torch.Tensor):
        if tensor.device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        if tensor.device.type == "cpu":
            return torch.autocast(device_type="cpu", enabled=False)
        from contextlib import nullcontext
        return nullcontext()

    @staticmethod
    def _summarize_tensor(tensor: torch.Tensor) -> dict:
        detached = tensor.detach()
        summary = {
            "shape": tuple(detached.shape),
            "dtype": str(detached.dtype),
            "is_contiguous": bool(detached.is_contiguous()),
            "stride": tuple(detached.stride()),
            "device": str(detached.device),
        }
        return summary

    def _run_mamba_token_mixer(self, seq: torch.Tensor) -> torch.Tensor:
        chunk_size = self.token_chunk_size
        if chunk_size <= 0 or seq.size(0) <= chunk_size:
            chunks = [seq]
        else:
            chunks = list(torch.split(seq, chunk_size, dim=0))

        mixed_chunks = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.contiguous()
            for block_idx, block in enumerate(self.mamba_token_mixer):
                try:
                    chunk = block(chunk)
                except RuntimeError as exc:
                    if "mamba_ssm is required" in str(exc):
                        raise
                    tensor_summary = self._summarize_tensor(chunk)
                    raise RuntimeError(
                        "Mamba token mixer failed at block "
                        f"{block_idx} chunk {chunk_idx} with seq={tensor_summary}"
                    ) from exc
            mixed_chunks.append(chunk)
        if len(mixed_chunks) == 1:
            return mixed_chunks[0]
        return torch.cat(mixed_chunks, dim=0)

    def _prepare_mixer_input(self, seq: torch.Tensor) -> torch.Tensor:
        if self.mamba_amp_policy == "fp32":
            return seq.float().contiguous()
        if self.mamba_amp_policy == "fp16":
            return seq.to(dtype=torch.float16).contiguous()
        if self.mamba_amp_policy == "bf16":
            return seq.to(dtype=torch.bfloat16).contiguous()
        return seq.contiguous()

    def forward(self, rgb_feat: torch.Tensor, spike_feat: torch.Tensor) -> torch.Tensor:
        if rgb_feat.dim() != 5:
            raise ValueError("mamba early fusion expects rgb with shape [B, T, 3, H, W].")
        if spike_feat.dim() != 5:
            raise ValueError("mamba early fusion expects spike with shape [B, T, S, H, W].")

        bsz, steps, rgb_chans, height, width = rgb_feat.shape
        spike_bsz, spike_steps, spike_bins, spike_height, spike_width = spike_feat.shape
        if (bsz, steps, height, width) != (spike_bsz, spike_steps, spike_height, spike_width):
            raise ValueError("rgb and spike must share batch, time, height, and width dimensions")
        if rgb_chans != 3:
            raise ValueError(f"Expected rgb channels=3, got {rgb_chans}")

        with self._profiled_timer("mamba_rgb_encoder"):
            rgb_flat = rgb_feat.reshape(bsz * steps, 3, height, width)
            rgb_low = self.rgb_context_encoder(rgb_flat)
            _, token_dim, token_h, token_w = rgb_low.shape
            rgb_low = rgb_low.reshape(bsz, steps, token_dim, token_h, token_w)

        with self._profiled_timer("mamba_spike_encoder"):
            spike_flat = spike_feat.reshape(bsz * steps * spike_bins, 1, height, width)
            spike_low = self.spike_token_encoder(spike_flat).reshape(bsz, steps, spike_bins, token_dim, token_h, token_w)

        with self._profiled_timer("mamba_token_pack"):
            spike_tokens = spike_low.permute(0, 1, 4, 5, 2, 3).reshape(
                bsz * steps * token_h * token_w, spike_bins, token_dim
            ).contiguous()
            rgb_tokens = rgb_low.permute(0, 1, 3, 4, 2).reshape(
                bsz * steps * token_h * token_w, 1, token_dim
            ).contiguous()
            seq = (spike_tokens + rgb_tokens).contiguous()
            mixer_input_dtype = seq.dtype

        with self._profiled_timer("mamba_mixer"):
            if self.mamba_amp_policy == "fp32":
                with self._device_autocast_disabled_context(seq):
                    seq = self._prepare_mixer_input(seq)
                    seq = self._run_mamba_token_mixer(seq)
            else:
                seq = self._prepare_mixer_input(seq)
                seq = self._run_mamba_token_mixer(seq)
            seq = seq.to(dtype=mixer_input_dtype)

        with self._profiled_timer("mamba_writeback"):
            pooled = seq.mean(dim=1).reshape(bsz, steps, token_h, token_w, token_dim).permute(0, 1, 4, 2, 3)
            fused_low = pooled + rgb_low
            writeback = self.fusion_writeback_head["body"](fused_low.reshape(bsz * steps, token_dim, token_h, token_w))
            delta_low = self.fusion_writeback_head["delta"](writeback)
            gate_logits_low = self.fusion_writeback_head["gate"](writeback)

        with self._profiled_timer("mamba_upsample"):
            delta = F.interpolate(delta_low, size=(height, width), mode="bilinear", align_corners=False).reshape(bsz, steps, 3, height, width)
            gate_logits = F.interpolate(gate_logits_low, size=(height, width), mode="bilinear", align_corners=False).reshape(bsz, steps, 3, height, width)
        gate = torch.sigmoid(gate_logits)
        effective_update = self.alpha.view(1, 1, 3, 1, 1) * gate * delta
        out = rgb_feat + effective_update

        if self.enable_diagnostics:
            self._last_diagnostics = {
                "token_norm": float(spike_tokens.detach().float().norm(dim=-1).mean().item()),
                "mamba_norm": float(seq.detach().float().norm(dim=-1).mean().item()),
                "delta_norm": float(delta.detach().float().abs().mean().item()),
                "gate_mean": float(gate.detach().float().mean().item()),
                "effective_update_norm": float(effective_update.detach().float().abs().mean().item()),
                "warmup_stage": self._warmup_stage,
            }
        else:
            self._last_diagnostics = {"warmup_stage": self._warmup_stage}
        return out


__all__ = ['MambaFusionOperator']
