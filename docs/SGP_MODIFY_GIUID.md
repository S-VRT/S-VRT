## 0. 先判断：这种“缝合 SGP 替换 self-attention”是否符合 supervisor 预期？

### supervisor 预期（从 VRT 设计逻辑推断）

VRT 的核心在于：**同一层里，mutual attention 负责对齐/融合，self-attention 负责特征提取**；并且通过“shift”实现跨 clip 交互。([PubMed][1])
所以 supervisor 说“把 self-attention layer 替换成 SGP”更像是：**只动 self-attn（feature extraction），别动 mutual attn（alignment/fusion）**——这是最符合原论文职责划分的替换方式。

### 你们 task 的合理性与风险

TriDet/SGP 的动机是：self-attention 在视频特征上可能出现“**rank loss / instant discriminability 下降**”，于是用**多粒度时域聚合**替代全局 token mixing 来缓解。([paperswithcode.com][2])
把它迁移到 S-VRT（RGB+Spike）里，如果你们观察到 self-attn 后**通道/特征趋同导致信号辨识度下降**，那做这个 ablation 是合理的。

**但要批判性强调一个关键点：**

* TriDet 的 SGP 本质上是针对**1D 时间序列特征**设计的（temporal feature pyramid），而 VRT 的 self-attn 是在**时空 window tokens**上做的（时空混合更强）。
* 所以“直接把 window self-attn 原地替换成 1D SGP”如果不重新定义 token 轴，容易出现**错误的邻接关系**（把空间 flatten 后当时间卷积，语义是错的）。

👉 因此，**正确的实现必须明确：SGP 在 VRT 里到底沿哪个维度做聚合。**下面我给你一个“最稳、语义最正确、改动最小”的落地方案：**按空间位置做“逐像素/逐patch 的时间序列 SGP”**（只做 temporal mixing，不乱做 spatial flatten）。

---

## 1. 替换策略（写死，不留选择）

### 1.1 只替换哪些层？

按你提供的 cursor 总结文件（你贴的“迁移分析”）的意图：

* **只替换纯 self-attention 子层**（mut_attn=false 的分支）
* **mutual attention 保持原 WindowAttention**（不要用 SGP 替 mutual）

这与 VRT 的职责划分一致：mutual 做对齐融合，self 做特征提取。([PubMed][1])

### 1.2 不替换哪些层？

* 所有用于 motion/alignment 的 mutual-attention（以及其依赖的 shift/clip partition 逻辑）不动。
* 任何显式依赖 window partition 的互注意力路径不动（否则对齐崩）。

---

## 2. **你们在 VRT 里的"正确语义版本" 模块定义**

### 2.0 实现原则：完整拷贝 + Wrapper 适配

**重要：你们要实现的是完整拷贝 TriDet 的原版 SGPBlock 代码，然后写一个 wrapper 类做张量形状适配，而不是重新实现 SGP 逻辑。**

具体步骤：
1. **完整拷贝** `TriDet/libs/modeling/blocks.py` 中的 `SGPBlock` 类（179-299行）到你们的项目中
2. **创建 SGPWrapper 类**：将 VRT 的 `[B,D,H,W,C]` 张量转换为 SGPBlock 期望的 `[B,C,T]` 格式，调用原版 SGPBlock，然后转换回原格式
3. **保持原版 SGPBlock 的所有参数和内部逻辑不变**，只在 wrapper 层做张量重塑适配

> **SGPWrapper：一个适配器类，将 VRT 的 [B,D,H,W,C] 张量转换为 TriDet SGPBlock 期望的 [B,C,T] 格式，调用原版 SGPBlock，然后转换回原格式。**

### 2.1 输入输出形状（强约束）

在 VRT/S-VRT 的 attention 子层里，统一规定 self-only 分支输入输出为：

* **输入** `x`: `[B, D, H, W, C]`

  * B: batch
  * D: 帧数/时间长度
  * H, W: 空间尺寸（特征图分辨率，不一定是原图）
  * C: embedding/channel

* **输出** `y`: `[B, D, H, W, C]`（必须完全同形状）

> 如果你们代码内部是 channel-first（如 `[B, C, D, H, W]`），也可以，但必须在模块边界固定一种规范；下面我按 `[B,D,H,W,C]`写死，Cursor 需要在适配器里做 permute。

### 2.2 适配到 SGP 的"时序卷积"形状（写死）

SGPWrapper 内部通过张量重塑调用原版 SGPBlock，只允许沿 D 做 1D 操作：

1. LayerNorm（或你们 self-attn 原本的 norm）

* `x_norm`: `[B, D, H, W, C]`

2. 变形：把每个空间位置的时间序列抽出来

* `x_seq = reshape/permute(x_norm) -> [B*H*W, D, C]`

3. 再转成 Conv1D 习惯的 channel-first

* `x_conv = permute(x_seq) -> [B*H*W, C, D]`

4. 调用原版 SGPBlock：输入 `[B*H*W, C, D]`，输出同形状 `[B*H*W, C, D]`

5. 变回去

* `y_seq`: `[B*H*W, D, C]`
* `y`: `[B, D, H, W, C]`

6. residual（必须保留 self-attn 原 residual 语义）

* `out = x + DropPath(y)`（或你们原逻辑）

---

## 3. SGPBlock 内部结构（写死到“能照着复现”）

TriDet 的论文/代码把 SGP 描述为“多粒度聚合 + 解决 self-attn 的 rank loss/辨识度下降”，并用在特征金字塔里。([paperswithcode.com][2])
你们在 VRT 里实现一个**严格的 3-branch temporal SGP**（不要做更多变体）：

### 3.1 三分支多粒度（固定 3 个分支）

对 `x_conv: [N, C, D]`（N=B*H*W）做：

* Branch-0（细粒度）：`k=1` 的 depthwise Conv1D
* Branch-1（中粒度）：`k=sgp_k` 的 depthwise Conv1D
* Branch-2（粗粒度）：`k=sgp_w` 的 depthwise Conv1D

所有分支：

* stride = 1
* padding = “same”（保证长度 D 不变）
* groups = C（depthwise）
* 输出形状都为 `[N, C, D]`

> sgp_k、sgp_w 的默认值在 config 写死：例如 `sgp_k=3, sgp_w=7`（你也可以用你们现在 json 里已有默认，但必须在 docs 明确写死）。

### 3.2 门控融合（固定为“全局时序池化 → 两层 MLP → softmax(3)”）

门控权重只依赖输入，不依赖分支输出：

1. 全局描述：沿 D 做平均池化

* `g = mean(x_conv, dim=D)` → `[N, C]`

2. reduction（固定 ratio=sgp_reduction）

* `h = Linear(C -> C/sgp_reduction) + GELU` → `[N, C/r]`
* `logits = Linear(C/r -> 3)` → `[N, 3]`

3. softmax 得到权重

* `α = softmax(logits)` → `[N, 3]`

4. 融合（逐分支加权求和）

* `y = α0*B0 + α1*B1 + α2*B2` → `[N, C, D]`

### 3.3 输出投影（固定存在，避免“只 depthwise 不混通道”）

融合后必须加一个 **pointwise（1×1）Conv1D** 或 Linear 等价操作来做通道混合：

* `y = PWConv1D(C->C)`，输出 `[N, C, D]`

### 3.4 残差结构（固定，按 TriDet 原样）

TriDet 的 SGPBlock 在内部包含自己的 identity shortcut 与内部 FFN（即 `out += identity` 和随后的内部 FFN），因此推荐**保持 SGPBlock 原始内部残差结构**并在集成时把 SGP 当作自包含子层使用。也就是说：

- 在 repo 中保留 TriDet 原始的 `SGPBlock`（含内部 `out += identity` 与内部 FFN）。
- 在 `SGPWrapper` / 上层集成处，通过约定（例如 `use_inner=True`）或文档说明，**不要对 SGP 的输出再做一次外层 residual 或外层 FFN**。如果出于实现需要把 `use_inner` 设为 `False`（让外层处理残差），那也应明确记录并在集成时保证外层会执行与 TriDet 等价的 residual+FFN 行为。

这样可以严格对齐 TriDet 图示的 block 行为：SGP 自包含其内部 shortcut，外层不重复残差，从而避免“重复残差”导致的训练动力学混淆。

---

## 4. 在 S-VRT 里“替换 self-attn”的精确嵌入点（写死流程）

你需要让 Cursor 做的不是“到处替换”，而是**把 self-only 的 attention 子层改成调用 SGPWrapper**，其余保持原样。

### 4.1 TMSA / TMSAG / RTMSA 的替换规则（统一）

在每个 Transformer block 中（无论叫 TMSA/TMSAG/RTMSA）通常结构是：

* `x = x + Attn(LN(x))`
* `x = x + FFN(LN(x))`

你们要做的是：

* 若该 block 处于 **self-only 模式**（你们现在用 `mut_attn=false` 表示）：

  * `Attn` 替换为 `SGPWrapper`
* 若 `mut_attn=true`：

  * mutual attention 仍走原 WindowAttention
  * self-attn 子层是否替换？**写死为“不替换”**（先做最干净的 ablation：只测 self-only 层）

> 这样最符合 supervisor 的“替换 self-attention layer 试试”且不会破坏对齐路径。([PubMed][1])

### 4.2 “Stage 模块里 25% self-only 层”的处理

你给的 cursor 总结提到 Stage 的某个 residual_group 中有 25% self-only 层：

* 规则同上：**只要是 self-only，就用 SGPWrapper。**

### 4.3 所有调用签名必须兼容（减少改动面）

为了让 Cursor 不需要“到处改 forward 参数”，你要强制：

* `SGPWrapper.forward(x, *args, **kwargs)`

  * 只使用 `x`
  * 其余参数直接忽略（但保留接口，避免调用处爆炸）

---

## 5. 配置开关与默认值（写死）

在你们现有 json 体系里（你文件提到已经支持 `use_sgp`, `sgp_w`, `sgp_k`, `sgp_reduction`），把行为固定为：

* `use_sgp: true/false`：总开关
* `sgp_scope: "self_only"`：**写死只作用于 self-only 层**（不要提供更多 scope，避免决策空间）
* `sgp_k: 3`（默认）
* `sgp_w: 7`（默认）
* `sgp_reduction: 4`（默认）

并在 docs 明确说明：**sgp_w 必须 > sgp_k 且都为奇数**（same padding 才干净）。

---

## 6. 验收标准（Cursor 做完后你怎么判“缝合是对的”）

### 6.1 结构正确性（必须全部满足）

1. **shape 守恒**：所有替换点输出必须严格等于输入 shape `[B,D,H,W,C]`
2. **mutual attention 路径完全不变**（参数量、调用次数、输出 shape 都不变）
3. self-only block 的 attention FLOPs 应明显下降（至少从 O(N²) token-mixing 变为 O(D·C·(k+w)) 的 1D 卷积级别）

### 6.2 行为正确性（最小 smoke test）

* 随机输入跑通 forward/backward（你文件提到已有 smoke test 思路）
* 开 `use_sgp=false` 与当前 baseline 完全一致（bitwise 不一定，但至少数值误差在浮点容忍内；更现实的标准是：同 seed 下 loss 曲线趋势一致）

### 6.3 你们 task 相关的“信号辨识度”验证（写死一个可执行指标）

别只看 PSNR/SSIM；既然你的动机是“辨识度”，必须加一个**中间表征诊断**（不需要花哨）：

* 在替换的 self-only block 输出 `feat: [B,D,H,W,C]` 上：

  * 计算通道相关性矩阵的平均绝对相关（或 rank proxy，例如特征矩阵的有效秩）
  * 对比 baseline self-attn vs SGP：你期望 **SGP 的通道相关性更低 / 有效秩更高**（更“分散”）

这才真正对应 TriDet 提的“instant discriminability / rank loss”动机。([paperswithcode.com][2])

---

## 7. 结论：这算“正确缝合”吗？

**如果你按我上面这个“逐空间位置的 temporal SGP”去实现：**

* ✅ 语义上是对的：SGP 做时间聚合，不会把空间 flatten 后胡乱卷积
* ✅ 符合 supervisor 预期：你确实在“self-attention（特征提取）”的位置做了替换，同时保留 VRT 的 mutual attention 对齐机制([PubMed][1])
* ✅ 工程上风险可控：改动面小、shape 容易守恒、回滚简单

**反过来，如果 Cursor 当前的缝合是“把 window self-attn 的 token 序列 flatten 后直接喂给 1D SGP 卷积”**，那我会判定为：

* ❌ 语义大概率不正确（卷积邻接关系错）
* ❌ 即使跑通，也很难把效果变化归因到“辨识度改善”，因为你引入了结构性错误混合

---


[1]: https://pubmed.ncbi.nlm.nih.gov/38451763/?utm_source=chatgpt.com "VRT: A Video Restoration Transformer - PubMed"
[2]: https://paperswithcode.com/paper/tridet-temporal-action-detection-with?utm_source=chatgpt.com "TriDet: Temporal Action Detection with Relative Boundary Modeling | Papers With Code"
