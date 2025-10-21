# 0. 适用范围与版本说明（务必先读）

本开发指导文档固定实现**"RGB 主干 VRT + SpikeEncoder3D 并行分支 + 时间维最终 Cross‑Attention 融合"**的最简 Baseline。

## 0.1 技术路线

本 Baseline 严格对齐以下技术路线：

* **Spike 表征**：将曝光窗 [t₀,t₁] 内的脉冲流体素化为 **K×H×W**（默认 **K=32**）。
* **SpikeEncoder3D**：3D Conv 残差金字塔，沿**时间维**下采样到与 **VRT 编码端**各尺度一致的时域维度；各尺度输出**通道数**严格对齐 VRT。
* **融合位置与方式（关键变更）**：**完成各自时域建模后，用 Cross‑Attention 在时间维做最终融合**。
  - 公式：在每个尺度 i，`Ff_i = CrossAttn(Q=Fr_i, K=Fs'_i, V=Fs'_i)`
  - 其中 `Fr_i` 为 **RGB 经 VRT 的 TMSA+并行对齐**输出
  - `Fs'_i` 为 **Spike 经自分支 Temporal Self‑Attention**输出
* **Spike 分支新增**：为保证对称性与表征能力，Spike 分支在 3D 编码后，**仅做"时间维 Self‑Attention（逐像素位置）"**，不做对齐/光流/卷积门控等扩展。
* **解码端不变**：多尺度解码与跳连沿用 VRT 原实现，仅把输入特征由 `Fr_i` 改为 `Ff_i`。
* **损失**：**Charbonnier/L1 + VGG 感知**（不加入对抗/时序/频域损失）。
* **验收口径**：在相同数据与训练步数下，**PSNR/LPIPS 不低于「无 Spike」VRT**，且**运动边界锐度主观更好**。

## 0.2 与旧版本的差异

相比早期版本（Concat→1×1 于 TMSA 入口），**本版本的结构变更**如下：

* **融合位置变更**：从"进入每层 TMSA 之前 Concat→1×1"改为**"完成各自时域建模后，用 Cross‑Attention 在时间维做最终融合"**。
* **Spike 分支新增**：新增时间维 Self‑Attention 模块（`SpikeTemporalSA`），在 3D 编码后对 Spike 特征进行时序建模。
* **融合模块变更**：使用 `TemporalCrossAttnFuse` 替代 `Concat1x1PreTMSA`。

## 0.3 开发约束

> **本文档已移除所有"可选项"。开发人员不得做任何未在本文出现的改动**（包括但不限于：额外损失、额外注意力、额外正则、改变注意力维度、改变头数/激活/归一化等）。

> **常见坑必须规避**：曝光窗错位、Spike 计数未归一化导致梯度爆炸。

---

# 1. 代码目录（固定且必须遵守）

**统一采用 Fork 改造 VRT 官方仓库**的方式组织代码，避免重复造轮子。

```
repo_root/
├─ third_party/
│  ├─ VRT/                          # JingyunLiang/VRT fork（必须）
│  └─ SpikeCV/                      # Zyj061/SpikeCV（仅用于数据预处理/可视化）
├─ src/
│  ├─ data/
│  │  ├─ datasets/
│  │  │  ├─ spike_deblur_dataset.py    # 数据集定义（固定文件名）
│  │  │  └─ voxelizer.py               # 体素化（固定文件名）
│  │  └─ collate_fns.py
│  ├─ models/
│  │  ├─ spike_encoder3d.py            # SpikeEncoder3D（固定文件名）
│  │  ├─ spike_temporal_sa.py          # Spike 时间维 Self‑Attention（固定文件名）
│  │  ├─ fusion/
│  │  │  └─ cross_attn_temporal.py     # 最终融合 Cross‑Attention（固定文件名）
│  │  └─ integrate_vrt.py              # 与 VRT 对接（固定文件名）
│  ├─ losses/
│  │  ├─ charbonnier.py                # Charbonnier 损失（固定文件名）
│  │  └─ vgg_perceptual.py             # VGG 感知损失（固定文件名）
│  ├─ train.py                         # 训练入口（固定文件名）
│  └─ test.py                          # 测试入口（固定文件名）
├─ configs/
│  └─ deblur/
│     └─ vrt_spike_baseline.yaml       # 唯一配置（固定文件名）
├─ scripts/
│  ├─ setup_env.sh                     # 环境安装脚本
│  ├─ setup_env.ps1                    # Windows 环境安装脚本
│  ├─ prepare_data.py                  # 数据预处理脚本
│  ├─ launch_train.sh                  # 训练启动脚本（Linux）
│  └─ launch_train.ps1                 # 训练启动脚本（Windows）
└─ outputs/                            # 训练与测试产出目录（固定结构）
   ├─ logs/
   ├─ ckpts/
   ├─ visuals/
   └─ metrics/
```

> **强制要求**：
> - 删除旧版 `src/models/fusion/concat_1x1_pre_tmsa.py` 及任何调用；仅保留并使用 `fusion/cross_attn_temporal.py`。
> - **禁止修改**上述文件名与层级；如需新增，仅可在同级新增，不得重命名或移动已有文件。

---

# 2. 环境与依赖（固定且必须遵守）

## 2.1 基础环境要求

* **OS**：Linux 或 Windows（建议 Linux）
* **Python**：**3.10**
* **PyTorch**：**≥ 2.2**
* **CUDA**：**11.8 或 12.1**（与显卡驱动匹配即可）

## 2.2 环境安装脚本

运行 `scripts/setup_env.sh`（Linux）或 `scripts/setup_env.ps1`（Windows），**不得改动内容**。

### Linux 版本（scripts/setup_env.sh）

```bash
set -e
conda create -y -n vrtspike python=3.10
conda activate vrtspike
# PyTorch（按需切换 cu118 / cu121）
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 通用依赖
pip install opencv-python einops timm yacs pyyaml tqdm tensorboard lpips scipy scikit-image
# VGG 感知需要的 torchvision 已随上步装好
# BasicSR（VRT 依赖）
pip install basicsr
# 克隆子模块（保持固定路径）
mkdir -p third_party && cd third_party
# 请在浏览器中先 fork 再 clone；此处仅说明固定目录结构
# git clone https://github.com/<your_fork>/VRT.git VRT
# git clone https://github.com/Zyj061/SpikeCV.git SpikeCV
```

### Windows 版本（scripts/setup_env.ps1）

```powershell
conda create -y -n vrtspike python=3.10
conda activate vrtspike
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python einops timm yacs pyyaml tqdm tensorboard lpips scipy scikit-image
pip install basicsr
```

> 如需编译算子（VRT 某些版本需要），严格遵循 VRT 官方 README，不在此文档重复。

---

# 3. 数据组织、对齐与体素化（刚性约束）

## 3.1 目录规范（固定）

将数据统一组织为如下格式（必须一致）：

```
DATA_ROOT/
├─ train/
│  ├─ seq_xxx/
│  │  ├─ blur/        # 输入模糊帧，命名为 000000.png, 000001.png, ...
│  │  ├─ sharp/       # GT 清晰帧，对齐命名
│  │  ├─ spike/       # 原始 Spike 事件/脉冲文件（原始格式，按序号对应）
│  │  └─ spike_vox/   # 体素化后的缓存（由 prepare_data.py 生成）
│  └─ ...
└─ val/
   └─ seq_yyy/
      ├─ blur/
      ├─ sharp/
      ├─ spike/
      └─ spike_vox/
```

**固定约束**：`DATA_ROOT/{train,val}/seq_xxx/{blur,sharp,spike,spike_vox}` 目录结构不得更改。

**禁止**使用自定义命名；如原始数据不兼容，先运行 `scripts/prepare_data.py` 统一整理。

## 3.2 曝光窗与帧对齐（必须对齐）

* **时间窗对齐**：每个模糊帧 `blur/??????.png` 必须记录 `[t0,t1]`，Spike 子序列严格截取同窗。
* **对齐日志**：对齐日志写入 `outputs/logs/align_*.txt`，逐条记录：`frame_idx, t0, t1, event_count`。
* **对齐验证**：`prepare_data.py` 需产生完整的对齐日志以供验证。

> 若数据集未提供绝对时间戳，按数据集给出的"每帧起止"或"相对 tick"一致性处理；**禁止**随意插值或放缩时间轴。

## 3.3 体素化（固定实现，不得改）

* **K=32**（等分 `[t0,t1]`，计数→`log1p`→按像素通道标准化）
* **归一化**：`mean/std` 由 `prepare_data.py` 统计并写入配置
* **缓存格式**：体素缓存为 `.npy` 文件

### 核心实现（必须遵守）

`src/data/datasets/voxelizer.py`：

```python
import numpy as np

def voxelize(events, t0, t1, H, W, K=32):
    """
    将事件流体素化为 K 个时间段的计数图
    
    Args:
        events: (N, 3/4) -> (t, y, x) 或 (t, y, x, p)
        t0, t1: 时间窗口起止
        H, W: 图像高宽
        K: 体素段数
    
    Returns:
        vox: (K, H, W) 体素化计数
    """
    vox = np.zeros((K, H, W), dtype=np.float32)
    dur = (t1 - t0)
    bin_idx = np.floor((events[:,0] - t0) / max(dur, 1e-9) * K).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, K-1)
    ys = events[:,1].astype(np.int64)
    xs = events[:,2].astype(np.int64)
    np.add.at(vox, (bin_idx, ys, xs), 1.0)
    return vox  # 后续再做 log1p 与标准化
```

> 若原始 Spike 为二值触发序列，`events` 即触发记录；若为计数图，先展开为事件列表或直接按时间段累加计数。

### 体素缓存目录结构

```
DATA_ROOT/train/seq_xxx/spike_vox/
  000000.npy   # shape: (K,H,W)
  000001.npy
  ...
```

缓存由 `scripts/prepare_data.py` 统一生成，并在末尾输出均值方差统计到：`configs/deblur/vrt_spike_baseline.yaml` 的 `DATA.NORM.MEAN/STD` 字段。

---

# 4. DataLoader 约束（固定）

## 4.1 返回结构（固定）

`src/data/datasets/spike_deblur_dataset.py`：**只支持**下述返回结构：

```python
{
  'blur': FloatTensor[B, T, 3, H, W],      # RGB 输入（与 VRT 保持一致的 clip 组织方式）
  'sharp': FloatTensor[B, T, 3, H, W],     # GT 清晰帧
  'spike_vox': FloatTensor[B, T, K, H, W], # 体素化 Spike
  'meta': { 
    'seq': str, 
    'frame_idx': List[int], 
    't0': List[float], 
    't1': List[float] 
  }
}
```

## 4.2 数据约束

* **返回字典键名固定**：`blur, sharp, spike_vox, meta`
* **张量形状**：
  - `blur, sharp ∈ [B,T,3,H,W]`
  - `spike_vox ∈ [B,T,K,H,W]`
  - `T=5`（clip 长度固定为 5）
* **图像尺寸**：
  - 训练裁剪 `256×256`
  - 验证全尺寸
  - 短边 **≥ 256**
* **样本处理**：**不满足对齐/尺寸**的样本**丢弃并记日志**
* **Collate 失败**：对齐缺失或尺寸不一致必须**丢弃样本并记录日志**

---

# 4.5 架构演进对比（理解关键变更）

为帮助理解本版本的核心架构变更，以下展示从旧版到新版的架构演进：

## 旧版架构（Concat→1×1 融合）

```
输入模糊帧 (blur)           输入脉冲体素 (spike_vox)
      ↓                              ↓
  VRT 编码器                    SpikeEncoder3D
      ↓                              ↓
    Fr_i                           Fs_i
      ↓                              ↓
      └──────→ Concat ←──────────────┘
                 ↓
            1×1 Conv 降维
                 ↓
              Fused_i
                 ↓
            VRT TMSA
                 ↓
            VRT 解码器
                 ↓
           输出清晰帧 (recon)
```

**特点**：
- 融合位置：进入 TMSA **之前**
- 融合方式：通道维 Concat + 1×1 卷积
- Spike 分支：仅 3D 卷积编码，无时序建模

## 新版架构（Cross-Attention 融合）

```
输入模糊帧 (blur)                    输入脉冲体素 (spike_vox)
      ↓                                       ↓
  VRT 编码器                             SpikeEncoder3D
      ↓                                       ↓
  VRT TMSA                                  Fs_i
      ↓                                       ↓
    Fr_i                           时间维 Self-Attention
      ↓                                       ↓
      │                                     Fs'_i
      │                                       │
      └─────→ Cross-Attention ←──────────────┘
              (Q=Fr_i, K/V=Fs'_i)
                     ↓
                   Ff_i
                     ↓
               VRT 解码器
                     ↓
              输出清晰帧 (recon)
```

**特点**：
- 融合位置：完成各自时域建模**之后**
- 融合方式：时间维 Cross-Attention（Q 来自 RGB，K/V 来自 Spike）
- Spike 分支：3D 编码 + 时间维 Self-Attention
- RGB 分支：完整保留 VRT 的 TMSA 能力

## 关键差异总结

| 对比维度 | 旧版 | 新版 |
|---------|------|------|
| 融合时机 | TMSA **之前** | TMSA **之后** |
| 融合维度 | 通道维（Concat） | 时间维（Attention） |
| Spike 时序建模 | ❌ 无 | ✅ 有（Temporal Self-Attention） |
| RGB 时序建模 | 融合后才做 | 融合前已完成（VRT TMSA） |
| 信息流向 | 对称融合 | RGB 为主（Query），Spike 辅助（Key/Value） |
| 参数量 | 较少（仅 1×1 Conv） | 较多（两组 Attention） |

---

# 5. 模型实现（关键变化）

## 5.1 SpikeEncoder3D（必须实现）

**文件**：`src/models/spike_encoder3d.py`

### 功能与约束

* **输入**：`(B,T,K,H,W)`；以 `(B·T,K,H,W)` 形式经 `Conv3D+Res3D` 金字塔
* **时间下采样**：仅**在时间维**下采样以**匹配 VRT 编码各尺度的时间长度**
* **通道对齐**：各尺度通道数 **严格等于** VRT 对应尺度 `C_enc[i]`
* **输出**：列表 `Fs_i ∈ [B,T_i,C_i,H_i,W_i]`（i=1..L）

### 结构约束

* `conv3d( K→C0, k=3,s=1,p=1 )` + ReLU
* `res3d × 2`
* `down_t`：`conv3d( C0→C1, k=(3,3,3), s=(2,1,1), p=1 )`  // **只在时间维下采样**
* `res3d × 2`
* 重复到与 VRT 编码端各尺度**时间长度**一致（例如 T:5 → 5/3/2/1，按 VRT 实际层数匹配）
* 各尺度输出通道数严格等于 VRT 对应尺度 `C_enc[i]`（在配置中写死）

> **禁止**使用 Transformer/Cross-Attn 等扩展；SpikeEncoder3D 仅由 **Conv3D + 残差** 组成。

## 5.2 Spike 分支：时间维 Self‑Attention（新增，必须实现）

**文件**：`src/models/spike_temporal_sa.py`

### 功能与目的

* **目的**：在**每个空间位置 (h,w)** 上，仅沿**时间维 T_i**做自注意力，得到 `Fs'_i`
* **形状约定**：输入 `Fs_i ∈ [B,T_i,C_i,H_i,W_i]` → 重排为 `X ∈ [B·H_i·W_i, T_i, C_i]`

### 模块结构

* **模块堆叠**：每尺度 **1 个** Block（不得增删）
* **每个 Block 结构**：
  1. **Pre‑LN**（LayerNorm 在 `C_i` 维）
  2. **MultiheadAttention**（`embed_dim=C_i`，`num_heads=4`，`dropout=0`，`batch_first=True`）
  3. 残差相加
  4. **前馈 MLP**：`Linear(C_i→2·C_i)` → GELU → `Linear(2·C_i→C_i)`
  5. 残差相加
* **输出**：重排回 `[B,T_i,C_i,H_i,W_i]` 作为 `Fs'_i`

### 严格约束

> **禁止**添加位置编码、卷积替代、门控等任何变体；头数固定为 4；dropout 固定为 0；Block 数每尺度固定为 1。

## 5.3 RGB 分支：VRT 编码 + TMSA（不变）

* 严格复用 VRT 官方实现，得到 `Fr_i ∈ [B,T_i,C_i,H_i,W_i]`
* 其中已包含并行对齐与 TMSA 的效果
* **不得**改动 VRT 原始算子实现

## 5.4 最终融合：时间维 Cross‑Attention（新增，必须实现）

**文件**：`src/models/fusion/cross_attn_temporal.py`

### 功能与目的

* **目的**：以 **RGB 分支输出 `Fr_i` 为 Query**，以 **Spike 分支输出 `Fs'_i` 为 Key/Value**，在**时间维**对齐融合，输出 `Ff_i`

### 形状与流程（逐尺度 i 执行）

1. 输入：`Fr_i, Fs'_i ∈ [B,T_i,C_i,H_i,W_i]`
2. 重排：
   - `Q = [B·H_i·W_i, T_i, C_i]` 来自 `Fr_i`
   - `K,V = [B·H_i·W_i, T_i, C_i]` 来自 `Fs'_i`
3. **MultiheadAttention**（`embed_dim=C_i`，`num_heads=4`，`dropout=0`，`batch_first=True`）计算 `Y = Attn(Q,K,V)`
4. **前馈 MLP**（与 5.2 相同）+ 残差
5. 重排回 `[B,T_i,C_i,H_i,W_i]`，作为该尺度解码输入 `Ff_i`

### 严格约束

* **不做通道升降维**；`C_i` 必须与 VRT 期望一致
* **不引入**额外 1×1
* **不引入**额外门控/融合系数

## 5.5 与 VRT 集成（唯一允许的对接点）

**文件**：`src/models/integrate_vrt.py`

### 初始化

* 构造 `SpikeEncoder3D`
* 构造 `SpikeTemporalSA`（按尺度维持一个共享配置，内部按尺度自动适配 `T_i,C_i,H_i,W_i`）
* 构造 `TemporalCrossAttnFuse`（按尺度实例化）

### 前向流程

1. 通过 VRT 编码与 TMSA 得到 `Fr_1..L`
2. 通过 `SpikeEncoder3D` 得到 `Fs_1..L`
3. 逐尺度执行 `Fs'_i = SpikeTemporalSA(Fs_i)`
4. 逐尺度执行 `Ff_i = TemporalCrossAttnFuse(Fr_i, Fs'_i)`
5. 将 `Ff_1..L` 送入 **原 VRT 解码端**，其余逻辑保持不变

### 严格约束

> **严禁**改动 VRT 内部层定义、窗口大小、对齐策略或解码器结构；所有新增逻辑仅在上述 3 个模块内实现，并在 `integrate_vrt.py` 串联。

> 集成点以函数封装，便于后续消融；**严禁**在 VRT 内部随意修改张量形状与通道数。

---

# 6. 损失与优化（固定）

## 6.1 损失函数

* **Charbonnier Loss**：`CharbonnierLoss(delta=1e-3)`，默认对 `recon` 与 `sharp` 逐帧求和取均值
* **VGG Perceptual Loss**：`VGGPerceptualLoss(layers=['relu3_3'])`，权重默认为 **0.1**
* **总损失**：`L = Charbonnier(recon, sharp) + 0.1 · VGGPerceptual(recon, sharp)`

## 6.2 优化器与调度

* **优化器**：`AdamW(lr=2e-4, betas=(0.9,0.99), weight_decay=1e-4)`
* **调度器**：Cosine，Warmup 5k 步
* **AMP**：开启（PyTorch autocast + GradScaler）

---

# 7. 配置文件（唯一且必须完整）

**文件**：`configs/deblur/vrt_spike_baseline.yaml`

```yaml
SEED: 123
DATA:
  ROOT: "/abs/path/TO/DATA_ROOT"
  TRAIN_SPLIT: "train"
  VAL_SPLIT: "val"
  CROP_SIZE: 256
  CLIP_LEN: 5
  K: 32
  NORM:
    MEAN: 0.0   # 由 prepare_data.py 写入
    STD: 1.0    # 由 prepare_data.py 写入
MODEL:
  VRT_CFG: "third_party/VRT/options/deblur/vrt_base.yaml"
  CHANNELS_PER_SCALE: [96, 96, 96, 96]  # 与 VRT 对齐（示例）
  LAYERS: 4               # 尺度数 L，需与 VRT 匹配
  SPIKE_TSA:
    HEADS: 4
  FUSE:
    TYPE: "TemporalCrossAttn"
    HEADS: 4
TRAIN:
  BATCH_SIZE: 8
  EPOCHS: 80
  OPTIM:
    LR: 2.0e-4
  SCHED:
    WARMUP_STEPS: 5000
LOG:
  TENSORBOARD: true
  SAVE_DIR: "outputs"
```

> **不得**添加任何未列字段；不得改动 HEADS 与 LAYERS 以外的模型超参。

> **禁止**引入对抗/时序/频域等额外 loss 或模块；确保 Baseline 最小闭环。

---

# 8. 训练与验证（流程固定）

## 8.1 启动命令

### Linux

```bash
bash scripts/launch_train.sh
```

**`scripts/launch_train.sh` 内容**（不要改动）：

```bash
set -e
conda activate vrtspike
python -u src/train.py --config configs/deblur/vrt_spike_baseline.yaml \
  2>&1 | tee outputs/logs/train_$(date +%Y%m%d_%H%M%S).log
```

### Windows

```powershell
pwsh scripts/launch_train.ps1
```

## 8.2 训练脚本关键流程（必须遵守）

`src/train.py` 必须按以下顺序执行：

1. **固定随机种子与 cudnn 行为**：设定随机种子、cudnn benchmark=false、deterministic=true
2. **构建数据集/数据加载器**：启用 `pin_memory=True`, `num_workers≥8`
3. **构建模型**：VRT 主干 + SpikeEncoder3D + SpikeTemporalSA + TemporalCrossAttnFuse
4. **加载优化器/调度器/AMP**
5. **逐步训练**：每 step 计算 `L_charb + 0.1*L_vgg`
6. **验证与保存**：每 `1000` step 验证一次，保存 `best_psnr` 与 `last`
7. **日志记录**：TensorBoard 记录学习率、总损失、PSNR/LPIPS
8. **可视化**：每 `10k` step 保存三列图（`blur/recon/sharp`）到 `outputs/visuals/`

## 8.3 验证指标与对照要求

### 指标口径

* **PSNR**：RGB/全图与 Y 通道各一份
* **LPIPS**：感知相似度指标
* **验证约束**：验证时禁止随机裁剪与增广；验证全尺寸

### 对照要求

在相同数据与训练步数下，**同时训练纯 VRT（无 Spike）** 与 **本 Baseline**，要求：

* `PSNR ≥ base − 0.05dB`
* `LPIPS ≤ base + 0.005`
* 运动边界可视化更锐（保存三列图：blur / recon / sharp）
* 运动边界主观清晰度**可见提升**（保存可视化对比）

---

# 9. 测试与导出（固定）

**文件**：`src/test.py`

## 9.1 测试流程

* 输入文件夹：`--input_blur <dir> --input_spike_vox <dir> --output <dir>`
* 逐帧（或按 clip）推理并写出 PNG
* **禁止**写入除 PNG/JPEG 以外的格式
* 记录平均耗时/帧与显存峰值到 `outputs/metrics/test_*.json`

---

# 10. 关键实现片段（可直接粘贴，不得改动结构）

## 10.1 Spike 时间维 Self‑Attention Block（逐像素、沿 T）

```python
# src/models/spike_temporal_sa.py
import torch
import torch.nn as nn

class TemporalSelfAttentionBlock(nn.Module):
    """
    Spike 分支时间维自注意力模块
    在每个空间位置 (h,w) 上，仅沿时间维 T 做 Self-Attention
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            batch_first=True, 
            dropout=0.0
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), 
            nn.GELU(), 
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, x):  # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # 重排为 [BHW, T, C] 以在时间维做注意力
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        
        # Self-Attention
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        
        # Feed-Forward
        y = self.mlp(self.ln2(x))
        x = x + y
        
        # 重排回 [B, T, C, H, W]
        x = x.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return x  # [B, T, C, H, W]
```

## 10.2 时间维 Cross‑Attention 融合（Q=Fr，K/V=Fs'）

```python
# src/models/fusion/cross_attn_temporal.py
import torch
import torch.nn as nn

class TemporalCrossAttnFuse(nn.Module):
    """
    时间维 Cross-Attention 融合模块
    Q 来自 RGB 分支 (Fr)，K/V 来自 Spike 分支 (Fs')
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=heads, 
            batch_first=True, 
            dropout=0.0
        )
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), 
            nn.GELU(), 
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, Fr, Fs):  # Fr, Fs: [B, T, C, H, W]
        B, T, C, H, W = Fr.shape
        
        # 重排为 [BHW, T, C]
        Q = Fr.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        K = Fs.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        V = K
        
        # Layer Norm
        Q = self.ln_q(Q)
        K = self.ln_kv(K)
        V = self.ln_kv(V)
        
        # Cross-Attention
        Y, _ = self.attn(Q, K, V, need_weights=False)
        X = Q + Y
        
        # Feed-Forward
        Y = self.ffn(self.ln_ffn(X))
        X = X + Y
        
        # 重排回 [B, T, C, H, W]
        X = X.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return X  # [B, T, C, H, W]
```

## 10.3 集成示意（按尺度循环）

```python
# src/models/integrate_vrt.py 片段（示意）
# enc_feats: List[Fr_i] from VRT encoder+TMSA
# spk_feats: List[Fs_i] from SpikeEncoder3D

Fr_list = enc_feats  # [L]

# Spike 分支时间维 Self-Attention
Fs_list = [tsa(Fs) for tsa, Fs in zip(self.spk_tsa_blocks, spk_feats)]  # Fs'_i

# 时间维 Cross-Attention 融合
Ff_list = [fuse(Fr, Fs) for fuse, Fr, Fs in zip(self.fuse_blocks, Fr_list, Fs_list)]

# 将 Ff_list 交回 VRT 解码端的对应入口
```

## 10.4 Charbonnier Loss

```python
# src/losses/charbonnier.py
import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier 损失函数"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps ** 2))
```

## 10.5 VGG Perceptual Loss

```python
# src/losses/vgg_perceptual.py
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGPerceptualLoss(nn.Module):
    """VGG 感知损失"""
    def __init__(self, layer='relu3_3'):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        # 截断至 relu3_3（索引按 torchvision 固定）
        self.slice = nn.Sequential(*list(vgg.children())[:16])
        for p in self.slice.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    
    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return torch.mean((self.slice(x) - self.slice(y))**2)
```

---

# 11. 日志、断点与复现（固定）

## 11.1 训练产出

* 每次训练生成 `outputs/ckpts/{exp}/best.pth` 与 `last.pth`
* 将 `git rev-parse HEAD`、配置文件全文、环境 `pip freeze` 写入 `outputs/logs/run_*.txt`

## 11.2 复现要求

* 固定 SEED=123
* 不启用 cudnn benchmark（`cudnn.benchmark=False`）
* 启用确定性模式（`cudnn.deterministic=True`）

---

# 12. 验收清单（必须逐条核对，不得漏项）

* [ ] `spike_vox` 已以 `log1p` + 训练集统计 `mean/std` 标准化
* [ ] 体素缓存 `.npy` 与对齐日志存在且通过随机抽查
* [ ] `Fs_i` 的时间长度 `T_i` 与 VRT 对应尺度完全一致
* [ ] SpikeEncoder3D 的**各尺度时间长度与通道数**与 VRT 编码端一一对应
* [ ] **SpikeTemporalSA** 仅在时间维、逐像素做 Self‑Attention，`heads=4`
* [ ] **TemporalCrossAttnFuse** 以 `Q=Fr_i`、`K/V=Fs'_i` 做时间维融合，`heads=4`
* [ ] 解码端输入为 `Ff_i`，未改动 VRT 其他模块
* [ ] 仅使用 **Charbonnier + 0.1·VGG**
* [ ] Baseline 指标满足"**不低于纯 VRT**"的约束（`PSNR ≥ base − 0.05dB`，`LPIPS ≤ base + 0.005`）
* [ ] 运动边界可视化主观更锐
* [ ] 提交完整日志/可视化/权重/配置

---

# 13. 常见错误与强制排查步骤

## 13.1 数据相关错误

1. **曝光窗错位**：
   - 随机抽取 20 个样本
   - 叠加 `spike_vox.sum(dim=2)` 与 `blur` 的运动方向
   - 肉眼核对是否一致

2. **梯度爆炸**：
   - 若 loss 出现 NaN，立刻检查 `std≈0`
   - 确认是否遗漏 `log1p`

## 13.2 模型相关错误

3. **时长不匹配**：
   - `SpikeEncoder3D` 的时间下采样未与 VRT 对齐
   - 逐尺度打印 `T_i` 并断言一致

4. **通道不匹配**：
   - `C_i` 与 VRT 不一致
   - 在初始化阶段 assert
   - 确保融合后通道数严格等于 VRT 期望通道

5. **注意力维度错误**：
   - 未在时间维展平（应为 `[BHW, T, C]`）
   - 检查重排代码

## 13.3 训练相关错误

6. **训练不收敛**：
   - 先禁用 AMP 验证稳定性
   - LR 降至 `1e-4` 复核
   - 关闭 AMP 排除溢出

7. **显存溢出**：
   - 确认 `heads=4`
   - 确认 `dropout=0`
   - 确认 `Block=1`
   - 确认 `B` 与 `CROP_SIZE` 不得超配

8. **验证口径混乱**：
   - 确认未开启随机裁剪与增广
   - 确认验证时使用全尺寸

9. **指标不升**：
   - 先关闭 AMP 排除溢出
   - 再降低 LR 至 `1e-4` 验证稳定性

---

# 14. 迁移改造步骤（从旧版到本版，必须照做）

如果您之前实现了旧版 Baseline（Concat→1×1 于 TMSA 入口），请按以下步骤迁移：

1. **删除旧融合模块**：
   - 删除 `src/models/fusion/concat_1x1_pre_tmsa.py` 与所有调用

2. **新增新模块**：
   - 新增 `src/models/spike_temporal_sa.py`（按 10.1 节代码粘贴）
   - 新增 `src/models/fusion/cross_attn_temporal.py`（按 10.2 节代码粘贴）

3. **修改集成代码**：
   - 在 `src/models/integrate_vrt.py` 中，插入 5.5 节的前向流程

4. **更新配置文件**：
   - 按第 7 章填好字段，确保包含 `SPIKE_TSA` 和 `FUSE` 配置

5. **重新预处理数据**：
   - 重新跑 `scripts/prepare_data.py` 确保 `mean/std` 写入新配置

6. **训练与评估**：
   - 训练与评估，输出验收材料

---

# 15. 里程碑与工时（参考）

* **D0–D2**：完成数据规整与体素缓存
* **D3–D5**：实现 SpikeEncoder3D、SpikeTemporalSA 与 TemporalCrossAttnFuse
* **D6–D7**：跑通训练闭环并完成与纯 VRT 的**严格对照**
* **D8**：提交验收材料（指标表、可视化、日志、权重）

---

# 16. 交付物（固定）

1. **代码仓**：目录与文件名完全匹配本文档
2. **配置**：`configs/deblur/vrt_spike_baseline.yaml`
3. **权重**：`outputs/ckpts/best.pth`
4. **日志**：`outputs/logs/*.txt` 与 TensorBoard 目录
5. **指标**：`outputs/metrics/val_*.json`（含 PSNR/SSIM/LPIPS）
6. **可视化**：`outputs/visuals/seq/frame_{blur,recon,sharp}.png` 三列对齐展示

> 以上各项**缺一不可**。任何交付物缺失或与本文差异均视为不合格交付。

---

# 附录 A：术语对照表

| 术语 | 说明 |
|------|------|
| VRT | Video Restoration Transformer（视频修复 Transformer） |
| TMSA | Temporal Mutual Self-Attention（时间互注意力） |
| Spike | 脉冲相机信号 |
| Voxel | 体素（时空离散化表示） |
| Cross-Attention | 交叉注意力（Q 来自一个分支，K/V 来自另一分支） |
| Self-Attention | 自注意力（Q/K/V 来自同一输入） |

---

# 附录 B：快速检查命令

```bash
# 检查环境
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 检查数据对齐
python scripts/prepare_data.py --check_only

# 检查模型通道匹配
python -c "from src.models.integrate_vrt import *; # 添加断言检查"

# 训练前 dry-run
python src/train.py --config configs/deblur/vrt_spike_baseline.yaml --dry_run
```

---

**文档版本**：v1.0（最终版）  
**生成日期**：2025-10-09  
**维护者**：研究团队  
**状态**：正式发布，严格执行


