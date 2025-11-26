#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified Spike Data Loader

This is a lightweight spike .dat file loader that doesn't depend on
the full SpikeCV library. It directly reads the binary .dat format.

Spike .dat format:
- Binary file containing spike events
- Each event is typically stored as (x, y, t, p) or similar format
- File format depends on the specific camera/dataset
"""

import numpy as np
from pathlib import Path

from SpikeCV.SpikeCV.spkProc.reconstruction.tfp import TFP
from SpikeCV.SpikeCV.spkData.load_dat import SpikeStream


def load_spike_dat(dat_file_path, spike_h=250, spike_w=400):
    """Load spike data from a .dat file.
    
    This is a simplified loader that reads raw spike data and accumulates
    it into a 3D matrix (T, H, W).
    
    Args:
        dat_file_path (str or Path): Path to the .dat file
        spike_h (int): Spike camera height
        spike_w (int): Spike camera width
    
    Returns:
        np.ndarray: Spike matrix of shape (T, H, W) where T is number of time bins
    """
    dat_file_path = Path(dat_file_path)
    
    if not dat_file_path.exists():
        raise FileNotFoundError(f"Spike file not found: {dat_file_path}")
    
    # Read binary data
    with open(dat_file_path, 'rb') as f:
        data = f.read()
    
    # File size in bytes
    file_size = len(data)
    
    # Try to infer format from file size
    # Common formats:
    # 1. Raw spike stream: each byte is a spike value (0 or 1)
    # 2. Compressed format: various encodings
    
    # For GoPro spike dataset, assume format is:
    # Time steps × Height × Width binary data
    
    # Calculate expected number of time steps
    # Assuming each spike is 1 byte (uint8)
    expected_pixels_per_frame = spike_h * spike_w
    
    if file_size % expected_pixels_per_frame == 0:
        # Data fits perfectly into (T, H, W) format
        num_timesteps = file_size // expected_pixels_per_frame
        
        # Reshape data
        spike_data = np.frombuffer(data, dtype=np.uint8)
        spike_matrix = spike_data.reshape((num_timesteps, spike_h, spike_w))
        
        return spike_matrix
    
    else:
        # Try alternative format: may have header or different structure
        # For now, just read what we can and pad/truncate
        spike_data = np.frombuffer(data, dtype=np.uint8)
        
        # Calculate how many complete frames we can get
        total_pixels = len(spike_data)
        num_complete_frames = total_pixels // expected_pixels_per_frame
        
        if num_complete_frames == 0:
            # File too small, return zeros
            print(f"WARNING: Spike file too small ({file_size} bytes), returning zeros")
            return np.zeros((10, spike_h, spike_w), dtype=np.uint8)
        
        # Take only complete frames
        usable_pixels = num_complete_frames * expected_pixels_per_frame
        spike_data = spike_data[:usable_pixels]
        spike_matrix = spike_data.reshape((num_complete_frames, spike_h, spike_w))
        
        return spike_matrix


def load_spike_dat_alternative(dat_file_path, spike_h=250, spike_w=400, num_bins=200):
    """Alternative loader that tries different file format assumptions.
    
    Args:
        dat_file_path (str or Path): Path to the .dat file
        spike_h (int): Spike camera height
        spike_w (int): Spike camera width
        num_bins (int): Number of time bins to use
    
    Returns:
        np.ndarray: Spike matrix of shape (num_bins, spike_h, spike_w)
    """
    dat_file_path = Path(dat_file_path)
    
    if not dat_file_path.exists():
        raise FileNotFoundError(f"Spike file not found: {dat_file_path}")
    
    with open(dat_file_path, 'rb') as f:
        data = f.read()
    
    file_size = len(data)
    
    # Method 1: Try reading as float32
    if file_size % 4 == 0:
        try:
            spike_data = np.frombuffer(data, dtype=np.float32)
            expected_size = num_bins * spike_h * spike_w
            
            if len(spike_data) >= expected_size:
                spike_matrix = spike_data[:expected_size].reshape((num_bins, spike_h, spike_w))
                return spike_matrix
        except:
            pass
    
    # Method 2: Try reading as uint8
    spike_data = np.frombuffer(data, dtype=np.uint8)
    expected_size = num_bins * spike_h * spike_w
    
    if len(spike_data) >= expected_size:
        spike_matrix = spike_data[:expected_size].reshape((num_bins, spike_h, spike_w))
        return spike_matrix
    else:
        # File too small, return zeros
        print(f"WARNING: Spike file size mismatch, expected at least {expected_size} bytes, got {file_size}")
        return np.zeros((num_bins, spike_h, spike_w), dtype=np.uint8)


def voxelize_spikes_tfp(spike_matrix, num_channels=1, device='cpu', half_win_length=20):
    """Voxelize spike stream using SpikeCV's TFP reconstruction.
    
    Args:
        spike_matrix (np.ndarray): Spike data (T, H, W).
        num_channels (int): Number of output channels (1 or 4).
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
    
    tfp = TFP(H, W, device)
    
    def _clamp_center(center_idx):
        return max(half_win_length, min(center_idx, T - half_win_length))
    
    if num_channels == 1:
        centers = [_clamp_center(T // 2)]
    elif num_channels == 4:
        segment_edges = np.linspace(0, T, num_channels + 1, dtype=int)
        centers = []
        for start, end in zip(segment_edges[:-1], segment_edges[1:]):
            centers.append(_clamp_center((start + end) // 2))
    else:
        raise ValueError(f"Unsupported num_channels: {num_channels}. Must be 1 or 4.")
    
    frames = []
    for center in centers:
        frame = tfp.spikes2frame(spike_matrix, key_ts=int(center), half_win_length=half_win_length)
        frames.append(frame.astype(np.float32) / 255.0)
    
    return np.stack(frames, axis=0).astype(np.float32)


class SpikeStreamSimple:
    """轻量封装 SpikeCV 官方 `SpikeStream`，用于离线 .dat 加载。"""
    
    def __init__(self, filepath, spike_h=250, spike_w=400, print_dat_detail=False):
        self.filepath = Path(filepath)
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.print_dat_detail = print_dat_detail
        
        # 复用 SpikeCV 的 SpikeStream 实现，避免重复造轮子
        self._spike_stream = SpikeStream(
            offline=True,
            filepath=str(self.filepath),
            spike_h=self.spike_h,
            spike_w=self.spike_w,
            print_dat_detail=self.print_dat_detail,
        )
        self._spike_matrix = None  # 惰性缓存
    
    def _ensure_spike_matrix(self):
        if self._spike_matrix is None:
            matrix = self._spike_stream.get_spike_matrix(flipud=False)
            # SpikeStream 返回 bool，我们保持下游期望的 uint8 格式
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


if __name__ == '__main__':
    # Test the loader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python spike_loader.py <path_to_spike.dat> [spike_h] [spike_w]")
        sys.exit(1)
    
    dat_file = sys.argv[1]
    spike_h = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    spike_w = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    
    print(f"Loading spike data from: {dat_file}")
    print(f"Expected dimensions: {spike_h}x{spike_w}")
    
    try:
        spike_matrix = load_spike_dat(dat_file, spike_h, spike_w)
        print(f"Loaded spike matrix shape: {spike_matrix.shape}")
        print(f"Data type: {spike_matrix.dtype}")
        print(f"Value range: [{spike_matrix.min()}, {spike_matrix.max()}]")
        
        # Test voxelization
        spike_voxel_1ch = voxelize_spikes_tfp(spike_matrix, num_channels=1)
        print(f"\nVoxelized (1 channel) shape: {spike_voxel_1ch.shape}")
        
        spike_voxel_4ch = voxelize_spikes_tfp(spike_matrix, num_channels=4)
        print(f"Voxelized (4 channels) shape: {spike_voxel_4ch.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
