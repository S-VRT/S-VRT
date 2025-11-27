# SGP 模块说明

## 1. 背景与目的

SGP（Scalable Granularity Perception）来自 TriDet（arXiv:2303.07347），通过「瞬时分支 + 多尺度窗口卷积分支」提供更高效的局部/全局建模能力。本项目将其作为可选注意力替代，专门用于 VRT 中的纯 self-attention 层，以降低注意力的 `O(N²)` 成本并增强局部结构建模。

## 2. 代码位置

| 文件 | 内容 |
| --- | --- |
| `models/sgp_vrt.py` | `SGP` 与 `SGPBlock` 的完整实现 |
| `models/network_vrt.py` | `TMSA`, `TMSAG`, `Stage`, `RTMSA` 中的 `use_sgp` 接口与调度逻辑 |
| `options/*.json` | 实验/训练配置，包含 `use_sgp`、`sgp_w`、`sgp_k`、`sgp_reduction` 等参数 |

## 3. 模块结构

### 3.1 SGP 层

```14:95:models/sgp_vrt.py
class SGP(nn.Module):
    def forward(self, x):
        # 瞬时分支：SE-MLP 门控 + 全连接
        x_mean = x.mean(dim=1)
        instant_gate = self.fc_instant_gate(x_mean).view(B_, 1, C)
        instant_main = self.fc_main(x)
        instant_out = instant_gate * instant_main

        # 窗口分支：深度可分 1D Conv
        x_seq = x.permute(0, 2, 1)
        conv_w_out = self.conv_w_main(x_seq)
        conv_kw_out = self.conv_kw_main(x_seq)
        window_gate = self.conv_w_gate(x_seq)
        window_out = window_gate.permute(0, 2, 1) * (conv_w_out + conv_kw_out).permute(0, 2, 1)

        return x + instant_out + window_out
```

- **瞬时分支 (Instant Branch)**：对窗口内 token 做均值 → SE 风格 MLP (`reduction` 控制压缩比) → 生成门控 `φ(x)`，与 `FC(x)` 相乘，捕捉通道级动态。
- **窗口分支 (Window Branch)**：两个深度可分 1D 卷积核（`sgp_w` 与 `sgp_w * sgp_k`）捕捉细粒度与大核上下文的混合感受野，再乘以门控 `ψ(x)`。
- **残差连接**：输出为 `x + instant_out + window_out`，与 Transformer block 接口一致。

### 3.2 SGPBlock

```98:131:models/sgp_vrt.py
class SGPBlock(nn.Module):
    def forward(self, x):
        return x + self.drop_path(self.sgp(self.norm(x)))
```

- `LayerNorm` → `SGP` → `DropPath` → 残差加和。
- `DropPath` 内置在 block 中，启用时外部 TMSA 不再重复添加。

## 4. 在网络中的嵌入方式

### 4.1 TMSA 层

```786:852:models/network_vrt.py
if self.use_sgp and not mut_attn:
    from .sgp_vrt import SGPBlock
    self.attn = SGPBlock(...)
    self.drop_path = nn.Identity()
else:
    self.attn = WindowAttention(...)
    self.drop_path = DropPath(...)
...
if hasattr(self.attn, 'mut_attn'):
    attn_windows = self.attn(x_windows, mask=attn_mask)
else:
    attn_windows = self.attn(x_windows)
```

- 仅在 `mut_attn=False`（纯 self-attention）时替换为 SGP，互注意力层保持原注意力逻辑。
- `forward_part1` 中自动跳过 `norm1` 与外部 `DropPath`，因为 `SGPBlock` 已包含这些操作。

### 4.2 TMSAG / Stage / RTMSA

- **Stage**：第二个 `residual_group`（25% self-only 层）可启用 SGP；第一个含互注意力的 `residual_group` 始终保留原注意力。
- **RTMSA**（Stage 8）以 `TMSAG(mut_attn=False)` 形式堆叠，因此可整体切换到 SGP，提升最终精炼阶段的局部建模能力。

## 5. 配置与参数

| 配置项 | 说明 | 默认 |
| --- | --- | --- |
| `use_sgp` | 是否让纯 self-attention 层改用 SGP | `false` |
| `sgp_w` | 窗口分支主卷积核大小 | `3` |
| `sgp_k` | 大核倍率，实际核大小 `sgp_w * sgp_k` | `3` |
| `sgp_reduction` | 瞬时分支 MLP 的通道压缩比 | `4` |

示例（`options/gopro_rgbspike_local_debug.json`）：

```json
"use_sgp": true,
"sgp_w": 3,
"sgp_k": 3,
"sgp_reduction": 4
```

开启后，所有 Stage 中的纯 self-attention 层以及 Stage 8 的 RTMSA 都会切换到 SGP，互注意力层保持 `WindowAttention`。

## 6. 训练/推理注意事项

- **内存与算力**：SGP 避免了 QK^T/Softmax 的 O(N²) 计算，窗口长度较大时显存更友好；卷积分支在大核配置下会略增算力，需要根据显存预算调整 `sgp_k`。
- **兼容性**：SGPBlock 与标准 Transformer block 接口一致，梯度检查点、DropPath、LayerNorm 等常用组件在 `network_vrt.py` 中已做兼容处理，无需额外改动。
- **调参策略**：
  - 若追求更多局部纹理复原，可提高 `sgp_k` 以增大等效感受野。
  - 若显存紧张，可同时降低 `window_size` 与 `sgp_w`，或仅在部分 Stage 启用（通过自定义 `use_sgp` 传入 Stage 构造函数实现）。
  - 可以与 `mul_attn_ratio` 联动，仅对特定 Stage 的 self-only 残差组开启 SGP，形成「互注意力 + 卷积注意力」混合。

## 7. 调试与验证建议

1. **模块单测**：在 `models/sgp_vrt.py` 中添加随机张量单元测试，确认 `SGPBlock` 保持张量形状 `(B_, N, C)` 不变。
2. **训练日志**：关注 `options` 中 `use_sgp` 开关对应的 wandb 运行，比较 loss 曲线、PSNR/SSIM 等指标。
3. **性能 Profiling**：使用 `torch.cuda.profiler` 或 `nsys` 对比启用/关闭 SGP 的时延与显存，便于确定最佳 `sgp_w/sgp_k` 组合。

---

如需进一步扩展（例如引入可学习 `sgp_k` 或与多尺度 attention 混合），建议在本文件中补充实验记录与设计思路，保持文档与代码一致。

