import torch
import torch.nn.functional as F


def spatial_correlation_sample(
    input1,
    input2,
    kernel_size=1,
    patch_size=1,
    stride=1,
    padding=0,
    dilation_patch=1,
):
    """Repo-local spatial correlation operator compatible with SCFlow usage.

    The implementation is vectorized in PyTorch rather than delegating to the
    external `spatial_correlation_sampler` package. It preserves the tensor
    contract used by SCFlow and returns `[B, patch_h, patch_w, out_h, out_w]`.
    """
    if input1.ndim != 4 or input2.ndim != 4:
        raise ValueError(
            f"spatial_correlation_sample expects 4D inputs [B,C,H,W], "
            f"got {tuple(input1.shape)} and {tuple(input2.shape)}"
        )
    if input1.shape != input2.shape:
        raise ValueError(
            f"spatial_correlation_sample expects matching input shapes, "
            f"got {tuple(input1.shape)} and {tuple(input2.shape)}"
        )
    if kernel_size <= 0 or patch_size <= 0 or stride <= 0 or dilation_patch <= 0:
        raise ValueError(
            "kernel_size, patch_size, stride, and dilation_patch must all be > 0."
        )
    if patch_size % 2 == 0:
        raise ValueError(
            f"spatial_correlation_sample expects an odd patch_size, got {patch_size}."
        )

    bsz, chans, height, width = input1.shape
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"Invalid correlation output size ({out_h}, {out_w}) for input "
            f"{tuple(input1.shape)} and kernel_size={kernel_size}, stride={stride}, padding={padding}."
        )

    input1_cols = F.unfold(
        input1,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
    )

    radius = patch_size // 2
    max_disp = radius * dilation_patch
    expanded_padding = padding + max_disp

    # First extract all kernel neighborhoods for input2 over the expanded grid.
    # Then gather the displacement patch in one more unfold, which avoids the
    # explicit Python loop over patch offsets used by the temporary version.
    input2_cols = F.unfold(
        input2,
        kernel_size=kernel_size,
        padding=expanded_padding,
        stride=stride,
    )
    expanded_h = (height + 2 * expanded_padding - kernel_size) // stride + 1
    expanded_w = (width + 2 * expanded_padding - kernel_size) // stride + 1
    input2_cols = input2_cols.view(
        bsz,
        chans * kernel_size * kernel_size,
        expanded_h,
        expanded_w,
    )

    displaced_cols = F.unfold(
        input2_cols,
        kernel_size=patch_size,
        dilation=dilation_patch,
        stride=1,
    )
    displaced_cols = displaced_cols.view(
        bsz,
        chans * kernel_size * kernel_size,
        patch_size * patch_size,
        out_h * out_w,
    )

    corr = (input1_cols.unsqueeze(2) * displaced_cols).sum(dim=1)
    return corr.view(bsz, patch_size, patch_size, out_h, out_w)
