<!-- 4eef5eb1-3ecf-400b-9969-cf5b0b789869 ec947d39-28a5-4009-b87a-1ec0fe3007e2 -->
# Integrate flash-attn into S-VRT (one-shot plan)

Overview

Replace the existing softmax-based attention implementation with `flash-attn` across all attention modules in `models/architectures/vrt` so the whole model uses Flash Attention for Q/K/V computations. The plan covers environment setup, code edits, handling relative position bias & masks, testing, benchmarking, and a safe rollback/fallback.

Files to change (essential)

- `models/architectures/vrt/attention.py` — replace `attention()` implementation in `WindowAttention` with Flash Attention calls and add import(s).
- `models/architectures/vrt/stages.py` — ensure `TMSA` / `RTMSA` call into `WindowAttention` unchanged; adjust only if masking/format changes are required.
- `models/architectures/vrt/vrt.py` — no core changes expected, but add feature-flag wiring if desired (e.g., `use_flash_attn` opt).
- `requirement.txt` (optional) — add `flash-attn` dependency (or document manual install steps in README).

Key implementation notes (concise, copyable)

1) Install

```bash
# Recommended: use the flash-attn package built for your CUDA + PyTorch
pip install flash-attn
# If building from source (CUDA >= 11.7 and matching PyTorch):
# git clone https://github.com/flash-attention/flash-attention.git
# cd flash-attention
# pip install -e .  # or follow repo README build steps
```

2) Minimal replacement for `attention()` (concept)

```python
# new imports
try:
    from flash_attn.flash_attention import flash_attention
except Exception:
    # fallback will be torch scaled_dot_product_attention
    flash_attention = None

def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
    # q,k,v shapes: (B, num_heads, N, head_dim)
    # convert from current shapes
    # current code uses q,k,v shaped (B_//?, self.num_heads, N, C_head)

    # compute bias from relative position if needed
    attn_bias = None
    if relative_position_encoding:
        # relative_position_bias: (N, N, num_heads)
        rpb = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)]
        rpb = rpb.reshape(N, N, -1).permute(2, 0, 1)  # (num_heads, N, N)
        attn_bias = rpb.unsqueeze(0)  # (1, num_heads, N, N)

    if mask is not None:
        # mask shaped for windows; convert to additive bias (0 for keep, -inf for mask)
        # mask: (nW, N, N) -> expand to (B//nW, nW, num_heads, N, N) earlier used
        # Create attn_bias_mask with zeros and -1e9 where masked
        mask_bias = mask.clone().unsqueeze(1) * -1e9  # (nW,1,N,N)
        # expand to batch in calling code or here
        if attn_bias is None:
            attn_bias = mask_bias
        else:
            attn_bias = attn_bias + mask_bias

    if flash_attention is None:
        # fallback: compute classic softmax attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(*x_shape)
        return out

    # use flash_attention: expects q,k,v with shape (B, num_heads, N, head_dim)
    out = flash_attention(q, k, v, attn_bias=attn_bias, dropout_p=0.0, causal=False)
    # out shape: (B, num_heads, N, head_dim) -> transpose/reshape similar to original
    out = out.transpose(1, 2).reshape(*x_shape)
    return out
```

Notes on integration details

- Shapes: the project currently computes `q,k,v` as `(batch_windows, num_heads, N, head_dim)` in `WindowAttention`; keep that layout and call `flash_attention(q,k,v, attn_bias=..., dropout_p=0.)`.
- Relative position bias: convert the stored table into `(1, num_heads, N, N)` additive bias and pass as `attn_bias`.
- Mask: convert window masks into additive bias (large negative value) and add to `attn_bias` before passing.
- Mutual attention (mut_attn) flow: two paths exist—both use the same `attention()` entry, so replacement is unified.
- CUDA & PyTorch versions: use a flash-attn build that matches your CUDA and PyTorch (test on one machine first).

Testing and validation

- Unit test: run existing tests in `tests/` (e.g., smoke tests) after integration.
- Numerical check: compare outputs (L2 / max abs) of a single attention layer between original and flash-attn for random input (seeded).
- Integration smoke: run `python main_test_vrt.py` or a single forward pass to ensure no shape/device errors.

Benchmarking

- Use a representative input (e.g., batch/window size used in training) and measure:
  - peak GPU memory (nvidia-smi) and training throughput (images/s or clips/s)
  - time per forward/backward step
- Compare baseline vs flash-attn results and report numbers.

Fallback & Rollback

- Add a runtime switch `opt['model'].get('use_flash_attn', True)` or environment var `USE_FLASH_ATTN=1`.
- If flash-attn is unavailable, fall back to original softmax attention with a logged warning.

Deliverables (what I'll produce if you accept the plan)

- Concrete edits for `attention.py` replacing `attention()` with flash-attn integration (copy-paste ready code block).
- A small test script `tests/test_flash_attn_repl.py` that compares outputs and measures time/memory.
- README snippet with install commands and troubleshooting tips.
- A short benchmark report template and commands to run them.

Todos (implementation tasks)

- id: install_flash_attn

content: Install flash-attn and verify CUDA/PyTorch compatibility

- id: add_dependency

content: Add optional `flash-attn` note to `requirement.txt` or docs

- id: replace_window_attention

content: Replace `WindowAttention.attention` with flash-attn call

- id: adapt_mask_and_bias

content: Convert relative position bias and mask to flash-attn `attn_bias`

- id: replace_tmsa_calls

content: Ensure `TMSA`/`RTMSA` use the updated `WindowAttention` path

- id: add_runtime_switch

content: Add `use_flash_attn` opt flag with safe fallback

- id: unit_test_and_smoke

content: Add unit test and run smoke tests for numerical parity

- id: benchmark_and_report

content: Run benchmarks and collect memory/time improvements

Estimated effort and risks

- Estimated dev time: 2–6 hours (depending on build issues and testing environment)
- Risk: flash-attn build failures if CUDA/PyTorch versions mismatch; relative-position bias shape bugs; subtle numerical differences

If you approve, I'll create the concrete editable plan and produce the patch-ready code snippets for `attention.py`, the test script, and the README instructions. Do you want me to proceed and generate the exact edit snippets now?

### To-dos

- [ ] Install flash-attn and verify CUDA/PyTorch compatibility
- [ ] Add optional `flash-attn` note to `requirement.txt` or docs
- [ ] Replace `WindowAttention.attention` with flash-attn call
- [ ] Convert relative position bias and mask to flash-attn `attn_bias`
- [ ] Ensure `TMSA`/`RTMSA` use the updated `WindowAttention` path
- [ ] Add `use_flash_attn` opt flag with safe fallback
- [ ] Add unit test and run smoke tests for numerical parity
- [ ] Run benchmarks and collect memory/time improvements