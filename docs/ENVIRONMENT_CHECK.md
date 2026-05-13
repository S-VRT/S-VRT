# Environment Check

Last Updated: 2026-04-25
Workspace: `/root/projects/S-VRT`
Host: `autodl-container-2ccrzp7syw-b90af12d`

## Summary

- OS: `Ubuntu 22.04.5 LTS`
- Kernel: `5.15.0-78-generic`
- CPU: `Intel(R) Xeon(R) Platinum 8470Q`
- CPU Topology: `2 sockets / 104 physical cores / 208 threads`
- RAM: `1.0 TiB`
- GPU: `4 x NVIDIA GeForce RTX 4090`
- GPU Memory: `49140 MiB per GPU`
- NVIDIA Driver: `580.105.08`
- `nvidia-smi` CUDA Version: `13.0`
- CUDA Toolkit: `13.0`
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
- Available at check time: `947 GiB`
- Swap: `0 B`

### Filesystem

- Workspace filesystem for `/` and `/root/projects/S-VRT`: `30G` total, `24G` used, `6.1G` available

## GPU And CUDA

- GPU count: `4`
- GPU model: `NVIDIA GeForce RTX 4090`
- Driver Version: `580.105.08`
- CUDA Version reported by `nvidia-smi`: `13.0`
- Total GPU Memory: `49140 MiB per GPU`
- Persistence Mode: `On` for all GPUs
- Active GPU Processes at check time: none

`nvidia-smi` snapshot:

```text
Sat Apr 25 11:03:32 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:16:00.0 Off |                  Off |
| 48%   30C    P8             32W /  450W |       0MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        On  |   00000000:5A:00.0 Off |                  Off |
| 48%   32C    P8             24W /  450W |       0MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 4090        On  |   00000000:D8:00.0 Off |                  Off |
| 48%   32C    P8             30W /  450W |       0MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 4090        On  |   00000000:D9:00.0 Off |                  Off |
| 49%   30C    P8             12W /  450W |       0MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
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
- Installed toolkit directory: `/usr/local/cuda-13.0`
- `nvcc` exists at `/usr/local/cuda/bin/nvcc`
- `nvcc` is not currently on `PATH`
- NVCC version: `V13.0.88`

`/usr/local/cuda/bin/nvcc --version`:

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

### Runtime Libraries

Detected from `ldconfig -p`:

- `libcudnn.so.9`
- `libcudnn_cnn.so.9`
- `libcudnn_ops.so.9`
- `libcudnn_adv.so.9`
- `libcudnn_graph.so.9`
- `libcudnn_heuristic.so.9`
- `libcudart.so.13`
- `libcublas.so.13`
- `libcublasLt.so.13`
- `libnccl.so.2`

This machine has the primary CUDA 13 runtime libraries, cuDNN 9, and NCCL available at the system level. Compatibility symlinks and CUDA 12 variants are also present.

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
logfire
matplotlib
snntorch
```

### Installed Project Runtime

The project virtual environment is present and populated. At check time:

- PyTorch: `2.11.0+cu130`
- `torch.version.cuda`: `13.0`
- `torch.cuda.is_available()`: `True`
- `torch.cuda.device_count()`: `4`

The virtual environment can see all four GPUs successfully.

## Compatibility Notes

- Driver, toolkit, and PyTorch are aligned on CUDA `13.0` on this machine.
- `nvcc` is available by absolute path but not exported on `PATH`; custom CUDA extension builds may need either an explicit `CUDA_HOME=/usr/local/cuda` or a `PATH` update.
- The workspace filesystem only has about `6.1G` free at the time of the check, which is tighter than the previous record and may be limiting for large checkpoints, dataset expansion, or build artifacts.
- `lsb_release` is not installed on this machine, so OS distribution details were taken from `/etc/os-release`.
