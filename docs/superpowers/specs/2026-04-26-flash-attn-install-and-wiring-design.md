# Flash Attention 安装与代码接入设计

## 环境

- GPU: RTX 4090 (sm_89)
- PyTorch: 2.11.0+cu130 (UV 环境)
- CUDA: 13.0
- CPU: 208 核
- flash-attn 2.8.3：无预编译 wheel，需源码编译

## 目标

1. 编译安装 flash-attn 2.8.3
2. 在 `models/architectures/vrt/attention.py` 的 `WindowAttention.attention()` 中实际使用 flash-attn

## 安装命令

```bash
MAX_JOBS=80 FLASH_ATTN_CUDA_ARCHS=89 uv pip install flash-attn --no-build-isolation
```

- `FLASH_ATTN_CUDA_ARCHS=89`：只编译 sm_89（RTX 4090），减少模板实例数
- `MAX_JOBS=80`：80 核并行，73 个编译单元几乎同时完成
- `--no-build-isolation`：复用 UV 环境中已有的 torch/CUDA

## 代码变更（仅 attention.py）

### 路径 1：自注意力（relative_position_encoding=True）

有 relative position bias，flash-attn 不支持任意 bias。

改用 `torch.nn.functional.scaled_dot_product_attention`：
- 将 relative position bias reshape 为 `(1, num_heads, N, N)`
- 若有 shift-window mask，reshape 为 `(B_, 1, N, N)` 后与 bias 相加
- 合并后作为 `attn_mask` 传入 SDPA → PyTorch 自动选 mem-efficient 内核

### 路径 2：互注意力（relative_position_encoding=False）

无 relative position bias，优先使用 flash-attn。

条件判断（按优先级）：
1. `self.use_flash_attn and mask is None and q.dtype in (torch.float16, torch.bfloat16)`
   → 使用 `flash_attn_func(q_, k_, v_, softmax_scale=self.scale)`
   → q/k/v 需从 `(B_, num_heads, N, head_dim)` transpose 为 `(B_, N, num_heads, head_dim)`
2. 否则 → `F.scaled_dot_product_attention(q, k, v, attn_mask=mask_reshaped, scale=self.scale)`

### 不需要改动的文件

- `models/architectures/vrt/stages.py`
- `models/architectures/vrt/vrt.py`
- `models/select_network.py`

`use_flash_attn` 标志已正确从顶层传递到 `WindowAttention`。

## 验证

安装后运行：
```bash
uv run python -c "from flash_attn import flash_attn_func; print('flash-attn OK')"
```

代码接入后运行现有测试：
```bash
uv run pytest tests/models/test_fusion_early_adapter.py -x -q
```
