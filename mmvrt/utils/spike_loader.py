from pathlib import Path
import numpy as np
from typing import Any

# Optional dependency on SpikeCV: try to import, fall back to lightweight handlers if missing.
try:
    from SpikeCV.SpikeCV.spkProc.reconstruction.tfp import TFP  # type: ignore
    from SpikeCV.SpikeCV.spkData.load_dat import SpikeStream  # type: ignore
    _HAS_SPIKECV = True
except Exception:
    _HAS_SPIKECV = False


def load_spike_dat(dat_file_path, spike_h=250, spike_w=400):
    dat_file_path = Path(dat_file_path)
    if not dat_file_path.exists():
        raise FileNotFoundError(f"Spike file not found: {dat_file_path}")
    with open(dat_file_path, 'rb') as f:
        data = f.read()
    file_size = len(data)
    expected_pixels_per_frame = spike_h * spike_w
    if file_size % expected_pixels_per_frame == 0:
        num_timesteps = file_size // expected_pixels_per_frame
        spike_data = np.frombuffer(data, dtype=np.uint8)
        spike_matrix = spike_data.reshape((num_timesteps, spike_h, spike_w))
        return spike_matrix
    else:
        spike_data = np.frombuffer(data, dtype=np.uint8)
        total_pixels = len(spike_data)
        num_complete_frames = total_pixels // expected_pixels_per_frame
        if num_complete_frames == 0:
            print(f"WARNING: Spike file too small ({file_size} bytes), returning zeros")
            return np.zeros((10, spike_h, spike_w), dtype=np.uint8)
        usable_pixels = num_complete_frames * expected_pixels_per_frame
        spike_data = spike_data[:usable_pixels]
        spike_matrix = spike_data.reshape((num_complete_frames, spike_h, spike_w))
        return spike_matrix


def load_spike_dat_alternative(dat_file_path, spike_h=250, spike_w=400, num_bins=200):
    dat_file_path = Path(dat_file_path)
    if not dat_file_path.exists():
        raise FileNotFoundError(f"Spike file not found: {dat_file_path}")
    with open(dat_file_path, 'rb') as f:
        data = f.read()
    file_size = len(data)
    if file_size % 4 == 0:
        try:
            spike_data = np.frombuffer(data, dtype=np.float32)
            expected_size = num_bins * spike_h * spike_w
            if len(spike_data) >= expected_size:
                spike_matrix = spike_data[:expected_size].reshape((num_bins, spike_h, spike_w))
                return spike_matrix
        except Exception:
            pass
    spike_data = np.frombuffer(data, dtype=np.uint8)
    expected_size = num_bins * spike_h * spike_w
    if len(spike_data) >= expected_size:
        spike_matrix = spike_data[:expected_size].reshape((num_bins, spike_h, spike_w))
        return spike_matrix
    else:
        print(f"WARNING: Spike file size mismatch, expected at least {expected_size} bytes, got {file_size}")
        return np.zeros((num_bins, spike_h, spike_w), dtype=np.uint8)


def voxelize_spikes_tfp(spike_matrix, num_channels=1, device='cpu', half_win_length=20):
    if spike_matrix.ndim != 3:
        raise ValueError(f"spike_matrix must be 3D (T, H, W), got shape {spike_matrix.shape}")
    T, H, W = spike_matrix.shape
    if T <= 2 * half_win_length:
        raise ValueError(f"Need more time steps ({T}) than twice the half window ({2 * half_win_length}) for TFP.")
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


if _HAS_SPIKECV:
    class SpikeStreamSimple:
        """Wrapper around SpikeCV SpikeStream for offline .dat loading (real implementation)."""

        def __init__(self, filepath, spike_h=250, spike_w=400, print_dat_detail=False):
            self.filepath = Path(filepath)
            self.spike_h = spike_h
            self.spike_w = spike_w
            self.print_dat_detail = print_dat_detail
            self._spike_stream = SpikeStream(
                offline=True,
                filepath=str(self.filepath),
                spike_h=self.spike_h,
                spike_w=self.spike_w,
                print_dat_detail=self.print_dat_detail,
            )
            self._spike_matrix = None

        def _ensure_spike_matrix(self):
            if self._spike_matrix is None:
                matrix = self._spike_stream.get_spike_matrix(flipud=False)
                self._spike_matrix = np.array(matrix, dtype=np.uint8, copy=True)
                if self.print_dat_detail:
                    print(f"Loaded spike data via SpikeStream: {self._spike_matrix.shape}")
                    print(f"  Value range: [{self._spike_matrix.min()}, {self._spike_matrix.max()}]")

        def get_spike_matrix(self, flipud=True):
            self._ensure_spike_matrix()
            spike_matrix = self._spike_matrix
            if flipud:
                spike_matrix = np.flip(spike_matrix, axis=1)
            return spike_matrix.copy()

        def get_block_spikes(self, begin_idx, block_len):
            self._ensure_spike_matrix()
            end_idx = min(begin_idx + block_len, self._spike_matrix.shape[0])
            return self._spike_matrix[begin_idx:end_idx].copy()
else:
    class SpikeStreamSimple:
        """Fallback minimal SpikeStreamSimple when SpikeCV is not available."""

        def __init__(self, filepath, spike_h=250, spike_w=400, print_dat_detail=False):
            self.filepath = Path(filepath)
            self.spike_h = spike_h
            self.spike_w = spike_w
            self.print_dat_detail = print_dat_detail

        def get_spike_matrix(self, flipud: bool = True):
            mat = np.zeros((100, self.spike_h, self.spike_w), dtype=np.uint8)
            if flipud:
                mat = mat[:, ::-1, :]
            return mat

        def get_block_spikes(self, begin_idx, block_len):
            mat = self.get_spike_matrix(False)
            end_idx = min(begin_idx + block_len, mat.shape[0])
            return mat[begin_idx:end_idx].copy()


def voxelize_spikes_tfp(spike_matrix: np.ndarray, num_channels: int = 4, device: Any = 'cpu', half_win_length: int = 20):
    """Minimal voxelization: aggregates time dimension into `num_channels` bins."""
    if spike_matrix is None:
        return np.zeros((num_channels, 250, 400), dtype=np.float32)
    T, H, W = spike_matrix.shape
    voxels = np.zeros((num_channels, H, W), dtype=np.float32)
    # Simple uniform binning
    bin_size = max(1, T // num_channels)
    for c in range(num_channels):
        start = c * bin_size
        end = min(T, (c + 1) * bin_size)
        if start >= end:
            continue
        voxels[c] = spike_matrix[start:end].sum(axis=0).astype(np.float32)
    # Normalize
    maxv = voxels.max() if voxels.size > 0 else 1.0
    if maxv > 0:
        voxels = voxels / maxv
    return voxels


