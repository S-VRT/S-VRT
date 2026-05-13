from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class TileBox:
    top: int
    left: int
    bottom: int
    right: int


def crop_box_to_tile(
    xyxy: tuple[int, int, int, int],
    tile: TileBox,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = xyxy
    left = max(x1, tile.left)
    top = max(y1, tile.top)
    right = min(x2, tile.right)
    bottom = min(y2, tile.bottom)
    if left >= right or top >= bottom:
        return None
    return (left - tile.left, top - tile.top, right - tile.left, bottom - tile.top)


def mask_intersects_tile(xyxy: tuple[int, int, int, int], tile: TileBox) -> bool:
    return crop_box_to_tile(xyxy, tile) is not None


def build_hann_window(height: int, width: int) -> torch.Tensor:
    row = torch.hann_window(height, periodic=False)
    col = torch.hann_window(width, periodic=False)
    return torch.outer(row, col).clamp_min(1e-6)


def stitch_weighted_tiles(
    canvas_shape: tuple[int, ...],
    tiles: Iterable[tuple[torch.Tensor, TileBox]],
) -> torch.Tensor:
    tile_items = list(tiles)
    if not tile_items:
        return torch.zeros(canvas_shape, dtype=torch.float32)

    first_tile = tile_items[0][0]
    accum_dtype = torch.float64 if first_tile.is_floating_point() else torch.float32
    canvas = first_tile.new_zeros(canvas_shape, dtype=accum_dtype)
    weights = first_tile.new_zeros(canvas_shape, dtype=accum_dtype)

    for tile_tensor, box in tile_items:
        height = box.bottom - box.top
        width = box.right - box.left
        if tile_tensor.shape[-2:] != (height, width):
            raise ValueError(
                f"tile tensor spatial shape {tuple(tile_tensor.shape[-2:])} "
                f"does not match tile box {(height, width)}"
            )
        if tile_tensor.shape[:-2] != canvas.shape[:-2]:
            raise ValueError(
                f"tile tensor prefix shape {tuple(tile_tensor.shape[:-2])} "
                f"does not match canvas prefix {tuple(canvas.shape[:-2])}"
            )

        window = build_hann_window(height, width).to(device=tile_tensor.device, dtype=accum_dtype)
        weighted = tile_tensor.to(dtype=accum_dtype) * window
        canvas[..., box.top : box.bottom, box.left : box.right] += weighted
        weights[..., box.top : box.bottom, box.left : box.right] += window

    return (canvas / weights.clamp_min(1e-6)).to(dtype=first_tile.dtype)
