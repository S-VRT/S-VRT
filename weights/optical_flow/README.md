# Optical Flow Checkpoints

Place optical-flow network checkpoints under this directory. Keep the
checkpoint paths in `options/*.json` aligned with the files you actually have
locally.

Recommended layout:

- `weights/optical_flow/scflow/dt10_e40.pth` for SCFlow.
- `weights/optical_flow/spynet/spynet_sintel_final-3d2a1287.pth` for SpyNet.

If a checkpoint is missing from the current working tree, look it up from the
Git remote history or release artifacts, then download/copy it back to the
same relative path above. For example, after fetching the remote refs you can
inspect whether the tracked file exists in a remote branch:

```bash
git fetch origin
git ls-tree -r origin/main -- weights/optical_flow
git ls-tree -r origin/baseline -- weights/optical_flow
```

Then restore the file from the branch that contains it, or download the
project-provided checkpoint artifact and place it in the matching subdirectory.

After restoring checkpoints, verify the training/test option file you plan to
run. For example, if you run `options/gopro_rgbspike_server.json`, check its
`netG.optical_flow` block and make sure the checkpoint path points to the
restored SCFlow file:

```json
"optical_flow": {
  "module": "scflow",
  "checkpoint": "weights/optical_flow/scflow/dt10_e40.pth",
  "params": {}
}
```

For any other config, check `netG.optical_flow.module` and
`netG.optical_flow.checkpoint`, then update that config's `checkpoint` value so
it points to the actual local checkpoint path under `weights/optical_flow/`.
