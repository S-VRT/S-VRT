#!/usr/bin/env python
"""Run a zero-train DCNv2/DCNv4 forward comparison for S-VRT.

Usage on the server:
  CUDA_VISIBLE_DEVICES=0 python docs/dcn_ab_forward_probe.py 2>&1 | tee -a docs/execution.log
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.select_dataset import define_Dataset
from models.select_network import define_G
from utils import utils_option as option


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DCNv2 and DCNv4 forward outputs without training.")
    parser.add_argument("--opt", default="options/gopro_rgbspike_server.json")
    parser.add_argument("--pretrain", default="weights/vrt/006_VRT_videodeblurring_GoPro.pth")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gt-size", type=int, default=None, help="Override dataset gt_size for this probe.")
    parser.add_argument("--reducer-index", type=int, default=2)
    parser.add_argument("--variants", nargs="+", default=["DCNv2", "DCNv4"])
    parser.add_argument(
        "--disable-rgb-normalize",
        action="store_true",
        help="Set dataset rgb_normalize to None before sampling.",
    )
    parser.add_argument("--no-flow", action="store_true", help="Do not pass L_flow_spike to the model.")
    return parser.parse_args()


def stamp(label: str) -> None:
    print(f"===== {label} {datetime.now().strftime('%F %T')} =====", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_phase_value(value: Any, phase_index: int = 1) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[phase_index]
    return value


def prepare_dataset_opt(opt: dict, gt_size: int | None, disable_rgb_normalize: bool) -> dict:
    ds_opt = copy.deepcopy(opt["datasets"]["train"])
    ds_opt["phase"] = "train"
    ds_opt["gt_size"] = int(gt_size if gt_size is not None else resolve_phase_value(ds_opt.get("gt_size", 96)))
    if "dataloader_batch_size" in ds_opt:
        ds_opt["dataloader_batch_size"] = int(resolve_phase_value(ds_opt["dataloader_batch_size"]))
    if disable_rgb_normalize:
        ds_opt["rgb_normalize"] = None
    return ds_opt


def partial_load(network: torch.nn.Module, path: str) -> tuple[int, list[str], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    bare = network.module if hasattr(network, "module") else network
    ckpt = torch.load(path, map_location="cpu")
    old = ckpt.get("params", ckpt)
    state = bare.state_dict()

    matched = {}
    missing_in_pretrain = []
    shape_mismatch = []
    for key, value in state.items():
        if key not in old:
            missing_in_pretrain.append(key)
        elif old[key].shape == value.shape:
            matched[key] = old[key]
        else:
            shape_mismatch.append((key, tuple(value.shape), tuple(old[key].shape)))

    state.update(matched)
    bare.load_state_dict(state, strict=True)
    return len(matched), missing_in_pretrain, shape_mismatch


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def build_input(sample: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if "L_rgb" in sample and "L_spike" in sample:
        lq = torch.cat([sample["L_rgb"], sample["L_spike"]], dim=1)
    else:
        lq = sample["L"]

    flow = sample.get("L_flow_spike")
    return lq.unsqueeze(0), sample["H"].unsqueeze(0), None if flow is None else flow.unsqueeze(0)


def prefix_counts(keys: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        prefix = key.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    stamp("DCN A/B FORWARD PROBE START")

    base = option.parse(args.opt, is_train=True)
    base["path"]["pretrained_netG"] = args.pretrain
    base["train"]["partial_load"] = True
    base["netG"]["restoration_reducer"]["type"] = "index"
    base["netG"]["restoration_reducer"]["index"] = args.reducer_index

    ds_opt = prepare_dataset_opt(base, args.gt_size, args.disable_rgb_normalize)
    print("opt:", args.opt)
    print("pretrain:", args.pretrain)
    print("sample_index:", args.sample_index)
    print("dataset gt_size:", ds_opt.get("gt_size"))
    print("dataset rgb_normalize:", ds_opt.get("rgb_normalize"))
    print("reducer:", base["netG"].get("restoration_reducer"))
    print("variants:", args.variants)

    dataset = define_Dataset(ds_opt)
    sample = dataset[args.sample_index]
    lq, gt, flow = build_input(sample)

    print("sample key:", sample.get("key"))
    print("L shape:", tuple(lq.shape))
    print("H shape:", tuple(gt.shape))
    print("flow shape:", None if flow is None else tuple(flow.shape))
    print("L min/max/mean:", float(lq.min()), float(lq.max()), float(lq.mean()))
    print("H min/max/mean:", float(gt.min()), float(gt.max()), float(gt.mean()))

    outputs: dict[str, torch.Tensor] = {}

    for dcn_type in args.variants:
        print(f"----- VARIANT {dcn_type} -----", flush=True)
        opt = copy.deepcopy(base)
        opt["netG"]["dcn_type"] = dcn_type
        opt["netG"]["dcn_apply_softmax"] = False
        opt["netG"]["use_flash_attn"] = False

        seed_all(args.seed)
        net = define_G(opt)
        bare = net.module if hasattr(net, "module") else net
        matched, missing, mismatches = partial_load(net, args.pretrain)

        pa_classes = sorted({type(m.pa_deform).__name__ for m in bare.modules() if hasattr(m, "pa_deform")})
        print("matched:", matched)
        print("missing_in_pretrain:", len(missing), prefix_counts(missing))
        print("first_missing:", missing[:30])
        print("shape_mismatch:", len(mismatches))
        print("first_shape_mismatch:", mismatches[:20])
        print("pa_deform classes:", pa_classes)
        print("conv_first.in_channels:", bare.conv_first.in_channels)
        print("backbone_in_chans:", bare.backbone_in_chans)

        device = next(bare.parameters()).device
        net.eval()
        with torch.no_grad():
            lq_device = lq.to(device)
            gt_device = gt.to(device)
            flow_device = None if flow is None or args.no_flow else flow.to(device)
            out = net(lq_device, flow_spike=flow_device) if flow_device is not None else net(lq_device)
            l1 = torch.mean(torch.abs(out - gt_device)).item()
            mse = torch.mean((out - gt_device) ** 2).item()
            p = psnr(out.clamp(0, 1), gt_device.clamp(0, 1))

        print("out shape:", tuple(out.shape))
        print("out min/max/mean:", float(out.min()), float(out.max()), float(out.mean()))
        print("L1:", l1)
        print("MSE:", mse)
        print("PSNR_clamped:", p)
        outputs[dcn_type] = out.detach().cpu()

    if len(outputs) >= 2:
        names = list(outputs)
        base_name = names[0]
        for other_name in names[1:]:
            diff = outputs[base_name] - outputs[other_name]
            print(f"===== {base_name}_vs_{other_name} output diff =====")
            print("diff mean_abs:", float(diff.abs().mean()))
            print("diff max_abs:", float(diff.abs().max()))
            print("diff rmse:", float(torch.sqrt(torch.mean(diff ** 2))))

    stamp("DCN A/B FORWARD PROBE END")


if __name__ == "__main__":
    main()
