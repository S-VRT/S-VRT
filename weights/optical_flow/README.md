Place optical flow checkpoints here. Recommended layout:

- `weights/optical_flow/spynet_sintel_final-3d2a1287.pth` (existing SpyNet checkpoint path)
- `weights/optical_flow/sea_raft.ckpt` (SEA-RAFT default checkpoint — add a pretrained checkpoint file here)

Notes:
- Large checkpoints should use Git LFS. For smoke tests we fall back to random initialization if a checkpoint is not provided or fails to load.


