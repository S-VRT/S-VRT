import numpy as np

from .reconstruction.tfp import TFP


def voxelize_spikes_tfp(spike_matrix, num_channels=1, device='cpu', half_win_length=20):
    """Voxelize spike stream using SpikeCV's TFP reconstruction.

    Args:
        spike_matrix (np.ndarray): Spike data (T, H, W).
        num_channels (int): Number of output channels. Supports any positive integer.
            For num_channels=1: uses center frame.
            For num_channels>1: evenly divides time sequence into segments and uses center of each segment.
        device (str or torch.device): Device for torch tensor ops used by TFP.
        half_win_length (int): Half window length for the TFP algorithm.

    Returns:
        np.ndarray: Voxelized spikes (num_channels, H, W) in float32 [0, 1].
    """
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be 3D (T, H, W), got shape {spike_matrix.shape}")

    T, H, W = spike_matrix.shape
    if T <= 2 * half_win_length:
        raise ValueError(
            f"Need more time steps ({T}) than twice the half window ({2 * half_win_length}) for TFP."
        )

    if num_channels <= 0:
        raise ValueError(f"num_channels must be positive, got {num_channels}")

    tfp = TFP(H, W, device)

    def _clamp_center(center_idx):
        return max(half_win_length, min(center_idx, T - half_win_length))

    if num_channels == 1:
        centers = [_clamp_center(T // 2)]
    else:
        segment_edges = np.linspace(0, T, num_channels + 1, dtype=int)
        centers = []
        for start, end in zip(segment_edges[:-1], segment_edges[1:]):
            centers.append(_clamp_center((start + end) // 2))

    frames = []
    for center in centers:
        frame = tfp.spikes2frame(spike_matrix, key_ts=int(center), half_win_length=half_win_length)
        frames.append(frame.astype(np.float32) / 255.0)

    return np.stack(frames, axis=0).astype(np.float32)
