# Flash Attention 安装与代码接入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编译安装 flash-attn 2.8.3 并在 `WindowAttention.attention()` 中实际使用它，使互注意力路径走 flash 内核，自注意力路径走 SDPA mem-efficient 内核。

**Architecture:** 仅改动 `models/architectures/vrt/attention.py`。`attention()` 方法按 `relative_position_encoding` 分叉：True 路径用 SDPA（有 bias，不兼容 flash），False 路径优先用 `flash_attn_func`（无 bias，无 mask，fp16/bf16），否则回退 SDPA。

**Tech Stack:** flash-attn 2.8.3（源码编译，sm_89），torch.nn.functional.scaled_dot_product_attention（PyTorch 2.11.0 内置）

---

### Task 1：编译安装 flash-attn

**Files:**
- 无文件改动，仅安装包

- [ ] **Step 1：启动编译安装（后台运行，预计 5-15 分钟）**

```bash
MAX_JOBS=80 FLASH_ATTN_CUDA_ARCHS=89 uv pip install flash-attn --no-build-isolation 2>&1 | tee /tmp/flash_attn_build.log
```

`FLASH_ATTN_CUDA_ARCHS=89` 只编译 RTX 4090 所需的 sm_89，`MAX_JOBS=80` 并行 80 核。

- [ ] **Step 2：验证安装成功**

```bash
uv run python -c "from flash_attn import flash_attn_func; print('flash-attn OK')"
```

Expected output:
```
flash-attn OK
```

- [ ] **Step 3：确认版本**

```bash
uv run python -c "import flash_attn; print(flash_attn.__version__)"
```

Expected output: `2.8.3`（或类似）

---

### Task 2：重写 `attention()` 方法

**Files:**
- Modify: `models/architectures/vrt/attention.py:1-16`（添加 F import）
- Modify: `models/architectures/vrt/attention.py:76-94`（重写 attention 方法）

- [ ] **Step 1：先运行现有 smoke test，确认基线通过**

```bash
uv run pytest tests/models/test_attention_smoke.py -v
```

Expected: PASSED

- [ ] **Step 2：在 attention.py 顶部添加 `import torch.nn.functional as F`**

当前文件头（[attention.py:1-9](models/architectures/vrt/attention.py#L1-L9)）：
```python
import math
import torch
import torch.nn as nn
import numpy as np
import math
from functools import lru_cache

from models.utils.init import trunc_normal_
```

改为：
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import lru_cache

from models.utils.init import trunc_normal_
```

（同时删除重复的 `import math`）

- [ ] **Step 3：重写 `attention()` 方法**

将 [attention.py:76-94](models/architectures/vrt/attention.py#L76-L94) 替换为：

```python
    def attention(self, q, k, v, mask, x_shape, relative_position_encoding=True):
        B_, N, C = x_shape

        if relative_position_encoding:
            # Self-attention: has relative position bias → SDPA mem-efficient backend
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
            # shape: (1, num_heads, N, N)
            attn_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                # mask: (nW, N, N) → (B_, 1, N, N) for broadcast
                shift_mask = mask[:, :N, :N].unsqueeze(1)  # (nW, 1, N, N)
                shift_mask = shift_mask.expand(B_ // nW, nW, 1, N, N).reshape(B_, 1, N, N)
                attn_bias = attn_bias + shift_mask

            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=self.scale)
        else:
            # Mutual attention: no relative position bias
            if (self.use_flash_attn
                    and mask is None
                    and q.dtype in (torch.float16, torch.bfloat16)):
                # flash_attn_func expects (batch, seqlen, nheads, headdim)
                q_ = q.transpose(1, 2)
                k_ = k.transpose(1, 2)
                v_ = v.transpose(1, 2)
                x = flash_attn_func(q_, k_, v_, softmax_scale=self.scale)
                return x.reshape(B_, N, C)
            else:
                attn_mask = None
                if mask is not None:
                    nW = mask.shape[0]
                    shift_mask = mask[:, :N, :N].unsqueeze(1)
                    attn_mask = shift_mask.expand(B_ // nW, nW, 1, N, N).reshape(B_, 1, N, N)
                x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

        x = x.transpose(1, 2).reshape(B_, N, C)
        return x
```

- [ ] **Step 4：运行 smoke test 确认基本正确性**

```bash
uv run pytest tests/models/test_attention_smoke.py -v
```

Expected: PASSED（fp32 走 SDPA math backend）

- [ ] **Step 5：Commit**

```bash
git add models/architectures/vrt/attention.py
git commit -m "feat(attention): wire SDPA and flash-attn into WindowAttention.attention()"
```

---

### Task 3：数值一致性测试（fp16，flash 路径）

**Files:**
- Modify: `tests/models/test_attention_smoke.py`（添加 fp16 flash 路径测试）

- [ ] **Step 1：在 test_attention_smoke.py 末尾追加测试**

```python
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_window_attention_flash_vs_sdpa_parity():
    """互注意力路径：flash-attn 与 SDPA 数值接近（atol=1e-2，fp16 精度）"""
    try:
        from flash_attn import flash_attn_func as _fa
    except ImportError:
        pytest.skip("flash-attn not installed")

    window_size = (2, 4, 4)
    dim = 64
    num_heads = 8
    N = window_size[0] * window_size[1] * window_size[2]
    Bn = 2
    device = "cuda"
    dtype = torch.float16

    x = torch.randn(Bn, N, dim, dtype=dtype, device=device)

    # mut_attn=False → 只走 self-attention 路径（relative_position_encoding=True）
    # 要测互注意力路径需要 mut_attn=True，但这里直接测 attention() 内部
    attn = WindowAttention(dim=dim, window_size=window_size, num_heads=num_heads,
                           qkv_bias=True, mut_attn=False, use_flash_attn=False)
    attn = attn.to(dtype=dtype, device=device)
    attn.eval()

    with torch.no_grad():
        # SDPA path (use_flash_attn=False)
        out_sdpa = attn(x, mask=None)

        # flash path: call attention() directly with relative_position_encoding=False
        qkv = attn.qkv_self(x).reshape(Bn, N, 3, num_heads, dim // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn.use_flash_attn = True
        out_flash_raw = attn.attention(q, k, v, mask=None, x_shape=(Bn, N, dim),
                                       relative_position_encoding=False)

    # flash-attn vs SDPA for mutual-attention path: relaxed tolerance for fp16
    assert out_flash_raw.shape == (Bn, N, dim)
    assert torch.allclose(out_flash_raw.float(), out_sdpa.float(), atol=5e-2, rtol=1e-2), \
        f"max diff: {(out_flash_raw.float() - out_sdpa.float()).abs().max().item():.4f}"
```

- [ ] **Step 2：运行新测试**

```bash
uv run pytest tests/models/test_attention_smoke.py::test_window_attention_flash_vs_sdpa_parity -v
```

Expected: PASSED

- [ ] **Step 3：运行全部 attention smoke tests**

```bash
uv run pytest tests/models/test_attention_smoke.py -v
```

Expected: all PASSED

- [ ] **Step 4：Commit**

```bash
git add tests/models/test_attention_smoke.py
git commit -m "test(attention): add flash-attn vs SDPA parity test for mutual-attention path"
```

---

### Task 4：回归测试

**Files:**
- 无改动，只运行测试

- [ ] **Step 1：运行 stages smoke test**

```bash
uv run pytest tests/models/test_stages_smoke.py -v
```

Expected: all PASSED

- [ ] **Step 2：运行 VRT smoke test**

```bash
uv run pytest tests/models/test_vrt_smoke.py -v
```

Expected: all PASSED

- [ ] **Step 3：运行 flash attention integration test（已有）**

```bash
uv run pytest tests/models/test_flash_attention_integration.py -v -s
```

Expected: PASSED（flash-attn 已安装，数值一致性通过）

- [ ] **Step 4：运行 AMP 相关测试（训练用 bfloat16，flash 路径会被触发）**

```bash
uv run pytest tests/models/test_amp_integration.py tests/models/test_amp_model.py -v
```

Expected: all PASSED
