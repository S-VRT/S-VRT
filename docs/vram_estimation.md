## VRT 训练显存测算（GoPro 去模糊配置）

### 1. 训练配置与假设
- `main_train_vrt.py` + `options/vrt/006_train_vrt_videodeblurring_gopro.json`
- DDP 3 卡，`dataloader_batch_size=3` → 每卡有效 batch=1，输入裁剪为 `6×3×224×224`
- dtype: FP32；Adam 优化器；`use_checkpoint_attn/ffn=true`（部分 stage 被白名单关闭）
- 模型：VRT (pa_frames=2，window=`[6,8,8]`，depths=`[8,8,8,8,8,8,8,4,4,4,4]`，embed_dims=`[96×7, 120×4]`)

### 2. 权重 + 优化器显存

| 模块 | 参数量 (M) | 权重占用 (MB) | 权重+梯度+Adam 状态 (MB) |
| --- | --- | --- | --- |
| SpyNet 光流 | 1.44 | 5.49 | 21.98 |
| conv_first | 0.023 | 0.09 | 0.36 |
| Stage1–7 (每个) | 2.03±0.04 | 7.74–7.88 | 30.9–31.5 |
| Stage8+norm+conv_after | 2.54 | 9.69 | 38.76 |
| Reconstruction (conv_last) | 0.003 | 0.01 | 0.04 |
| **总计** | **18.32** | **69.9** | **279.6** |

> 计算方式：`显存 ≈ 参数量 × 4 byte`。训练时需要存储权重、梯度、Adam 一阶/二阶动量 → `×4`。

### 3. 激活显存分解
以下数字全部按 **单卡 batch=1、FP32** 估算。表格中的 `Base` 是 stage 输出特征（会在 skip 连接中长时间存留），`QKV`/`Attn`/`FFN` 是每个 block 在前向峰值需要的瞬时显存。Gradient Checkpoint 会让大多数 block 只在前向瞬间分配 QKV/FFN，但 `no_checkpoint_*` 列表中的 stage 需要把这些张量一直保留到反向。

| Stage | 特征尺寸 (C×D×H×W) | Base (MB) | QKV (MB) | Attn (MB) | FFN (MB) | Checkpoint 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (scale×1) | 96×6×224×224 | 110.2 | 2 646 | 10 584 | 3 528 | attn/ffn checkpoint 开启 |
| 2 (×1/2) | 96×6×112×112 | 27.6 | 661.5 | 2 646 | 882 | ffn 关闭，其余 checkpoint |
| 3 (×1/4) | 96×6×56×56 | 6.9 | 165.4 | 661.5 | 220.5 | attn/ffn 均关闭 → 需常驻 |
| 4 (×1/8) | 96×6×28×28 | 1.7 | 54.0 | 216.0 | 55.1 | attn/ffn 均关闭 |
| 5 (×1/4) | 96×6×56×56 | 6.9 | 165.4 | 661.5 | 220.5 | attn/ffn 均关闭 |
| 6 (×1/2) | 96×6×112×112 | 27.6 | 661.5 | 2 646 | 882 | ffn 关闭，attn checkpoint |
| 7 (×1) | 96×6×224×224 | 110.2 | 2 646 | 10 584 | 3 528 | checkpoint 开启 |
| 8–11 (RTMSA，每层4 block，C=120) | 120×6×224×224 | 137.8 | 1 653.8 | 10 584 | 2 205 | 全部 checkpoint |

**计算依据示例（stage1，自注意力 block）：**
1. 窗口大小 `Wd×Wh×Ww = 6×8×8 = 384`，窗口数量 `nW = ceil(6/6)×ceil(224/8)^2 = 784`
2. QKV 投影：`3 × nW × WdWhWw × C = 3×784×384×96 = 1.73×10^8` elements → `661.5 MB`。互注意力窗口 `(2×8×8)` 贡献 `1 984.5 MB`，合计 `2 646 MB`
3. 注意力矩阵：`nW × heads × (WdWhWw)^2 = 784×6×384^2 → 5 292 MB`（互注意力同量），合计 `10 584 MB`
4. GEGLU FFN：需要同时保存两个 2×扩展通道 → `B×D×H×W×C×mlp_ratio×2 = 3.528 GB`

> 由于 Checkpoint，在 stage1/2/6/7/8–11 中，上述 QKV/FFN 张量只在正向单个 block 内存在。stage3–5 的 QKV 与 FFN、stage2/6/10 的 FFN 会在正向结束后依旧驻留，于反向时再次读取。

### 4. 其他显存开销
- **SpyNet 光流激活**：5 个层级，每级 `5 × 2 × H×W` 通道，总计 ~5.3 MB（长时间保留，用于后续 parallel warping）。
- **平行对齐 (DCNv2 + flow warp)**：`x_backward/x_forward` 各 `≈4×input ≈ 14.3 MB`，再加上 DCN 中的 offset/mask (~5 MB)。
- **Skip features**：`x1,x2,x3,x4` 会缓存直至 decoder 回传，总计 `~146 MB`。
- **数据/标签**：LQ & GT 各 `6×3×224×224×4 byte ≈ 3.6 MB`，加上 dataloader pinned buffer `~10 MB`。
- **CUDA runtime / cuDNN workspace**：约 0.8–1.0 GB，在多次 FFT/PixelShuffle 时会瞬时增大。

### 5. 单卡总显存预估

| 组成 | 估算值 |
| --- | --- |
| 权重 + 梯度 + Adam | ~0.28 GB |
| 长驻激活（skip、flows、非 checkpoint block） | ~0.20 GB |
| 单个大 stage 峰值（QKV + Attn + FFN + Base） | stage1/7 ≈ 16 GB；stage8–11 ≈ 14 GB；stage2/6 ≈ 4.2 GB |
| 其他 (warp buffers、输入、runtime) | ~1.0 GB |
| **峰值需求 (安全冗余 15%)** | **≈ 18–20 GB / GPU** |

因此在 3×DDP 下，需要 **≥20 GB** 显存的 GPU（例如 RTX 4090 / A5000 / A40）才能稳定完成 `224×224` patch 训练。若使用 16 GB 级别 GPU，可考虑：
1. 减小 `gt_size` 或 `num_frame`
2. 进一步打开 checkpoint（放宽 `no_checkpoint_*`，代价是算力×2）
3. 启用 AMP/FSDP（需额外工程投入）

### 6. 复盘与备注
- 无法直接在当前环境跑完整前向测峰值（GPU 已被其它进程占用，`torch.cuda.max_memory_allocated` 读数失败）。因此以上数值全部来自解析网络结构的解析式计算。
- 若需要实测，可在空闲 GPU 上运行 `scripts/profile_vram.py`（可仿照此处 Python 片段：对每个 stage 重置 `torch.cuda.reset_peak_memory_stats()` 并记录 `max_memory_allocated()`）。
- 链接：`models/network_vrt.py`（Stage/TMSA 定义）、`options/vrt/006_train_vrt_videodeblurring_gopro.json`（训练超参）、`main_train_vrt.py`（DDP 流程）。










