<!-- 7a163403-b55e-4c21-9e52-0f47221db0b7 bafca9d0-aaf7-4590-98a3-a7ab473e67e4 -->
# Modernize DDP and torchrun Support

## Overview

Transform the training code from legacy `torch.distributed.launch` to modern `torchrun` with automatic distributed environment detection.

## Key Changes

### 1. Update `utils/utils_dist.py`

- Modify `_init_dist_pytorch()` to read `LOCAL_RANK` from environment variables instead of requiring `RANK`
- Change device assignment to use `LOCAL_RANK` instead of `RANK % num_gpus`
- Ensure backward compatibility for both old and new launch methods

**Current issue**: Line 25 reads `RANK` which may not be set in older setups; line 27 uses `rank % num_gpus` instead of `LOCAL_RANK`.

### 2. Update `main_train_vrt.py`

- Replace argument parsing to remove `--local_rank` parameter dependency
- Add automatic distributed detection function that checks `WORLD_SIZE` environment variable
- Move distributed initialization before option parsing to set rank early
- Remove manual `--dist` flag requirement - auto-detect from `WORLD_SIZE > 1`

**Current issue**: Lines 39-46 use old-style argument parsing with manual `--dist` flag and unused `--local_rank`.

### 3. Update `utils/utils_option.py`

- Remove or make optional the `gpu_ids` field manipulation (lines 110-112)
- Stop setting `CUDA_VISIBLE_DEVICES` when in distributed mode (it conflicts with torchrun)
- Auto-detect `num_gpu` from `WORLD_SIZE` environment variable when in DDP mode
- Keep `gpu_ids` field for backward compatibility but don't enforce it

**Current issue**: Line 111 sets `CUDA_VISIBLE_DEVICES` which conflicts with torchrun's device assignment.

### 4. Update `models/model_base.py`

- In `__init__`, change device detection to use `LOCAL_RANK` when in distributed mode
- Modify `model_to_device()` to get device_ids from `LOCAL_RANK` instead of `current_device()`
- Ensure rank 0 checks remain consistent

**Current issue**: Line 12 uses simplistic device detection; line 107 uses `current_device()` which may not be set correctly.

### 5. Update launch script `launch_train.sh`

Replace single-process launch with torchrun command supporting both single and multi-GPU scenarios.

**New command format**:

```bash
torchrun --nproc_per_node=GPU_COUNT main_train_vrt.py --opt CONFIG_PATH
```

### 6. Update JSON config files

- Change `"dist": true` to be optional (auto-detected)
- Document that `"gpu_ids"` is informational only in DDP mode
- Add example for `"gpu_ids": "auto"`

## Implementation Order

1. Update utils_dist.py (foundation)
2. Update utils_option.py (config handling)
3. Update model_base.py (model device setup)
4. Update main_train_vrt.py (main entry point)
5. Update launch_train.sh (launcher script)
6. Update JSON config documentation

## Backward Compatibility

- Old configs with explicit `gpu_ids` will still work in single-GPU mode
- Manual `--dist` flag will be ignored (auto-detected instead)
- Legacy `torch.distributed.launch` will continue to work alongside torchrun

### To-dos

- [ ] Modify utils/utils_dist.py to use LOCAL_RANK env var and improve _init_dist_pytorch()
- [ ] Update utils/utils_option.py to not set CUDA_VISIBLE_DEVICES in distributed mode
- [ ] Modify models/model_base.py device initialization to use LOCAL_RANK properly
- [ ] Refactor main_train_vrt.py to auto-detect distributed mode and remove old args
- [ ] Update launch_train.sh to use torchrun instead of plain python
- [ ] Add comments to JSON config about auto-detection of DDP mode