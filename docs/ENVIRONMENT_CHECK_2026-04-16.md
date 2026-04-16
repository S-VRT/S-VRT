# Environment Check

Date: 2026-04-16
Workspace: `/root/projects/S-VRT`

## GPU And Driver

- GPU: `NVIDIA GeForce RTX 4080`
- Driver Version: `580.105.08`
- CUDA Version reported by `nvidia-smi`: `13.0`
- Total GPU Memory: `32760 MiB`
- Current GPU Processes: none

`nvidia-smi` snapshot:

```text
Thu Apr 16 11:26:40 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4080        On  |   00000000:9D:00.0 Off |                  N/A |
| 30%   31C    P8             14W /  320W |       0MiB /  32760MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## CUDA Toolkit

- `/usr/local/cuda` points to `/usr/local/cuda-12.8`
- `nvcc` exists at `/usr/local/cuda/bin/nvcc`
- `nvcc` is not currently in `PATH`
- CUDA Toolkit version: `12.8`
- NVCC version: `V12.8.93`

`nvcc --version` via absolute path:

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```

## Deep Learning Runtime Libraries

Detected from `ldconfig -p`:

- `libcudnn.so.9`
- `libcudnn_cnn.so.9`
- `libcudnn_ops.so.9`
- `libcudnn_adv.so.9`
- `libcudart.so.12`
- `libcublas.so.12`
- `libcublasLt.so.12`
- `libnccl.so.2`

This indicates the machine has the main CUDA runtime libraries, cuDNN 9, and NCCL installed at the system level.

## Compiler And Python

- GCC: `11.4.0`
- System Python: `3.12.3`
- Project `uv` virtual environment Python: `3.11.15`

## uv Environment Status

Actions performed:

- Created `.venv` with `uv venv --python 3.11 .venv`
- Confirmed `.venv/bin/python --version` is `Python 3.11.15`

Current sync status at the time of this note:

- `uv pip sync requirement.txt --python .venv/bin/python` is still running
- Observed process: `uv pip sync requirement.txt --python .venv/bin/python`
- The process is downloading large packages including:
  - `opencv-python`
  - `wandb`
  - `scikit-image`
  - `matplotlib`
  - `torchvision`
  - `pillow`
  - `tensorboard`
  - `timm`

Important note:

- `.venv` exists, but dependency installation had not finished when this document was written
- `.venv/lib/python3.11/site-packages` had not yet been populated with project dependencies at that point

## Compatibility Notes

- The machine is using a modern stack: driver supports CUDA `13.0`, local CUDA Toolkit is `12.8`
- `SpikeCV/requirements.txt` references much older packages such as:
  - `torch~=1.8.0+cu111`
  - `torchvision~=0.9.0+cu111`
- Those versions target the CUDA 11.1 era and may not have ready-made wheels for Python 3.11
- If this repository actually depends on those versions, PyTorch and CUDA compatibility should be verified before continuing with model training or custom CUDA extension builds
