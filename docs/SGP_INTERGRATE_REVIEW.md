你这张 **TriDet 论文 Figure 4** 其实把“SGP 到底替换了什么”说得非常死：它不是“把 Self-Attention 换成一个卷积模块”这么简单，而是**把一个 Transformer block 的前半段重构**：

* **Transformer Layer（原）**：`LN → Self-Attention → 残差加和 → LN → FFN → 残差加和`
* **SGP Layer（论文）**：`LN → SGP → 残差加和 → GN → FFN → 残差加和`
  也就是：**替换了 self-attention，且把“第二个 LN”换成 GN**（这是图注里明确写的）。([CVPR 2026][1])

而且图里 SGP 内部的“两个层级”不是摆设，它对应一个非常明确的设计意图：解决 self-attn 在视频特征上导致的 **rank loss / instant discriminability deterioration（特征趋同、可分性下降）**，并用不同时间尺度的卷积聚合替代“凸组合式”的 attention 聚合。([CVPR 2026][1])

---

## 1) 用 Figure 4 作为“判卷标准”，你们这次实现哪里算对、哪里不算

我按图里的结构逐项对照你们实现（`sgp.py` + 你上传的审查结论）来批判性判定：

### ✅ 方向对的点：你们确实做到了“只沿时间维做 SGP”（这很关键）

你们实现了把 `[B, D, H, W, C]` reshape 成 `[B*H*W, C, D]` 做 1D temporal mixing，再还原回去——这至少避免了“把 window flatten token 当时间”的经典语义错误（**如果训练时永远走 5D 分支**的话）。

> 这符合图里 SGP 的“temporal granularity”本意，而不是空间乱卷。([CVPR 2026][1])

---

## 2) 但如果你问“是否正确完成了 *TriDet 图里那种* SGP 缝合”：目前**结论是否定的**

原因不是“细节不一样”，而是**结构级别违背 Figure 4 的关键约束**——这会让实验结论无法干净归因，从 supervisor 的角度也很难算“按预期替换 self-attn”。

### ❌ (A) **残差结构错了：你们是双重残差**，而图里本质是“一个 SGP 残差块”

Figure 4 的 SGP Layer 只有一个清晰的残差路径：`x + f_SGP(LN(x))`，然后进入 `GN→FFN`。

但你们现在同时存在：

* SGP 模块内部 residual（`out += identity`）
* 外层 wrapper 又做 residual（`x_out = drop_path(x_sgp) + x_conv`）

这等价于把 block 变成一种非常规形式（类似 `x + LN(x) + f(LN(x))`），它已经不是“把 self-attn 换成 SGP”那么干净了。这个问题在你上传的审查文档里也被确认是**致命红旗**。

**为什么这会违背 supervisor 预期？**
supervisor 想看的通常是 **“同一 block 结构下，把 self-attn 算子换成 SGP 算子”** 的 ablation；双重残差会显著改变训练动力学，导致结果变化无法归因到 SGP 的“辨识度改善机制”。

---

### ❌ (B) **你们没有实现 Figure 4 明确要求的：第二个 LN → GN 的替换**

图注写得很明确：*replace the self-attention and the second LayerNorm (LN) with SGP and GroupNorm (GN), respectively.* ([CVPR 2026][1])

但你们当前实现里并没有把 “SGP 后、FFN 前”的 norm 换成 GN（审查文档也指出“无 GN/GroupNorm”）。

**为什么 GN 不是可有可无？**
TriDet 把“rank loss/趋同”这件事和 transformer 里的归一化行为绑在一起讨论（博客总结里也强调 LN 的几何性质会让特征模长固定、角度收缩，从而趋同更严重），所以他们才在 block 里动了第二个 norm。([九万里][2])
你们不换 GN，就不能声称“实现了 Figure 4 的 SGP layer”，最多只能说“做了一个 temporal conv mixer”。

---

### ❌ (C) **你们保留了一个极危险的 3D 分支：把 window token 的 N 当时间**

你们代码明确写了：`[B_, N, C]` 的 window format “treat N as temporal dimension”。审查文档也指出这会把 `window_size^2 × D` 这种时空混合 token 误当纯时间。

这点在 VRT/S-VRT 里尤其致命，因为 VRT 的窗口 token 序列 **不是纯时间轴**；一旦训练时走到这条分支，你的“语义正确的 temporal SGP”实验就被污染了。([CVPR 2026][1])

---

### ❌ (D) SGP 内部的“phi / psi 机制”与论文图的计算图不一致（所以别叫“认证版本”）

Figure 4 中：

* **Instant-level**：`AvgPool → FC + ReLU` 产生 **φ**，再去门控一个 `FC(x)` 分支
* **Window-level**：`Conv_w` 与 `Conv_kw`，并且有一个额外的 `Conv_w` 路径产生 **ψ** 做门控（图里标得很清楚）

你们现在实现的是“3 路卷积 + 产生 3 个 softmax 权重的 gate_net 再加权求和”，这更像“Mixture-of-Conv-Experts”，不是图里那种 φ/ψ 的双门控结构。

这不一定比论文差，但它已经是**新设计**，不能再对外说“基于 TriDet 的认证版本完整复现”。

---

## 3) 一句话判定

* **如果标准是**：“跑通、shape 守恒、给 self-only 路径塞进一个 temporal conv mixer”
  ✅ 你们大体做到了。

* **如果标准是**：“按 Figure 4 把 VRT 的 self-attention block 改造成 TriDet 的 SGP layer，来验证‘self-attn 导致辨识度下降’这条机制”
  ❌ 目前不算完成：**双残差 + 缺 GN + 3D 分支污染 + φ/ψ 不一致**，会让你的 ablation 失去可解释性。

---

## 4) 你接下来应该怎么做，才能“严格对齐 Figure 4”（给你一个很硬的验收清单）

你可以要求 Cursor 只做以下最小修复（不改功能边界，只修“是否符合图”）：

1. **按 TriDet 原始设计保留 SGP 内部残差与内部 FFN**：保留 SGPBlock 内部的 `out += identity` 和内部 FFN（即 SGPBlock 自包含其内部 shortcut 与 FFN）；在把 SGPBlock 嵌入 VRT 时，应确保外层不对 SGP 输出再加一层外部 residual 与外部 FFN（在代码中通过 `SGPBlock.use_inner=True` 或 SGPWrapper 的约定来保证）。换言之，集成时把 SGP 当作“完整的子层”使用，外层跳过重复的 `x + ...` 与外层 FFN。

2. **强制只接受 5D 输入**：删掉/禁用 `[B_, N, C]` 分支，防止 window token 被当时间。

3. **实现 “SGP 后、FFN 前”的 GN**：仅在启用 SGP 的 block 上，把原本 FFN 前的 LN 改为 GN（这就是 Figure 4 的“替换第二个 LN”）。([CVPR 2026][1])

4. **φ/ψ 按图实现**：至少在结构上做到：

   * `φ = ReLU(FC(AvgPool(x)))`，门控 `FC(x)` 分支
   * `ψ` 来自 window-level 的卷积分支输出（按图做一次门控）
   * 最终 `f_SGP = φ·FC(x) + ψ·(Conv_w(x)+Conv_kw(x)) + x`（这里的 `+ x` 是 SGP 内部的 shortcut，由 SGPBlock 自行实现）

做到这四条，才做的是 Figure 4 意义下的 SGP layer 替换 self-attn 的实验**

---
