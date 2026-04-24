# Environment Check

Last Updated: 2026-04-17
Workspace: `/root/projects/S-VRT`
Host: `autodl-container-233rvqugpm-0c6a446f`

## Summary

- OS: `Ubuntu 22.04.5 LTS`
- Kernel: `5.15.0-78-generic`
- CPU: `Intel(R) Xeon(R) Platinum 8470Q`
- CPU Topology: `2 sockets / 104 physical cores / 208 threads`
- RAM: `1.0 TiB`
- GPU: `NVIDIA GeForce RTX 4080`
- GPU Memory: `32760 MiB`
- NVIDIA Driver: `580.95.05`
- `nvidia-smi` CUDA Version: `13.0`
- CUDA Toolkit: `12.8`
- GCC: `11.4.0`
- System Python: `3.12.3`
- Project `.venv` Python: `3.11.15`
- `uv`: `0.11.7`
- PyTorch in `.venv`: `2.11.0+cu130`

## OS And Hardware

### OS

- Distribution: `Ubuntu 22.04.5 LTS (Jammy Jellyfish)`
- Kernel: `Linux 5.15.0-78-generic`
- Architecture: `x86_64`

### CPU

- Model: `Intel(R) Xeon(R) Platinum 8470Q`
- Sockets: `2`
- Cores per socket: `52`
- Threads per core: `2`
- Total logical CPUs: `208`
- NUMA nodes: `2`

### Memory

- Total RAM: `1.0 TiB`
- Available at check time: `928 GiB`
- Swap: `0 B`

### Filesystem

- Workspace filesystem for `/` and `/root/projects/S-VRT`: `30G` total, `22G` used, `8.3G` available

## GPU And CUDA

- GPU: `NVIDIA GeForce RTX 4080`
- Driver Version: `580.95.05`
- CUDA Version reported by `nvidia-smi`: `13.0`
- Total GPU Memory: `32760 MiB`
- Persistence Mode: `On`
- Active GPU Processes at check time: none

`nvidia-smi` snapshot:

```text
Fri Apr 17 15:13:58 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4080 ...    On  |   00000000:17:00.0 Off |                  N/A |
| 30%   28C    P8             13W /  320W |       0MiB /  32760MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

### CUDA Toolkit

- `/usr/local/cuda` exists and points into the system CUDA installation
- Installed toolkit directory: `/usr/local/cuda-12.8`
- `nvcc` exists at `/usr/local/cuda/bin/nvcc`
- `nvcc` is not currently on `PATH`
- NVCC version: `V12.8.93`

`/usr/local/cuda/bin/nvcc --version`:

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

### Runtime Libraries

Detected from `ldconfig -p`:

- `libcudnn.so.9`
- `libcudnn_cnn.so.9`
- `libcudnn_ops.so.9`
- `libcudnn_adv.so.9`
- `libcudnn_graph.so.9`
- `libcudnn_heuristic.so.9`
- `libcudart.so.12`
- `libcublas.so.12`
- `libcublasLt.so.12`
- `libnccl.so.2`

This machine has the primary CUDA runtime libraries, cuDNN 9, and NCCL available at the system level.

## Python And Project Runtime

### Base Toolchain

- GCC: `11.4.0`
- System Python: `3.12.3`
- `uv`: `0.11.7`
- `.venv` Python: `3.11.15`
- `.python-version`: not present

### Repository Dependency Declaration

Current `requirement.txt`:

```text
opencv-python
scikit-image
pillow
torchvision
hdf5storage
ninja
lmdb
requests
timm
einops
tensorboard
wandb
swanlab
matplotlib
snntorch
```

### Installed Project Runtime

The project virtual environment is present and populated. At check time:

- PyTorch: `2.11.0+cu130`
- `torch.version.cuda`: `13.0`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.device_count()`: `1`

This means the previous note claiming that dependency installation was still incomplete is no longer accurate for the current machine state.

## Compatibility Notes

- Driver support is newer than the locally installed CUDA toolkit, which is normal: driver reports CUDA capability `13.0`, while the installed toolkit is `12.8`.
- The project `.venv` is already using a CUDA 13 PyTorch build and can see the GPU successfully.
- `nvcc` is available by absolute path but not exported on `PATH`; custom CUDA extension builds may need either an explicit `CUDA_HOME=/usr/local/cuda` or a `PATH` update.
- The workspace filesystem only has about `8.3G` free at the time of the check, which may be tight for large checkpoints, dataset expansion, or build artifacts.
