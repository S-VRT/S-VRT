## VRT+Spike Baseline 实施进度

依据 `docs/Vrt+spike 视频去模糊 Baseline 开发指导.md` 严格执行，仅保留最简 Baseline 必要项。

### 里程碑清单
- [x] Scaffold 仓库结构、脚本、配置与进度文档
- [x] 实现体素化与数据预处理脚本（生成 .npy，统计 mean/std 并回写配置）
- [x] 实现 Spike 去模糊数据集与 collate 函数
- [x] 实现 Charbonnier 与 VGG 感知损失
- [x] ~~实现融合模块 Concat1x1PreTMSA~~（已废弃，迁移至新架构）
- [x] 实现 SpikeEncoder3D（多尺度 Conv3D 残差，支持仅时间维下采样）
- [x] ~~将 Spike 分支与 VRT 融合（TMSA 前逐尺度 Concat + 1x1x1 Conv3d）~~（已废弃）
- [x] **架构迁移至新版（Cross-Attention 融合）**
- [x] 实现 Spike 时间维 Self-Attention（spike_temporal_sa.py）
- [x] 实现时间维 Cross-Attention 融合（cross_attn_temporal.py）
- [x] 重构 integrate_vrt.py 以支持新架构
- [x] 训练入口（配置驱动、AMP、验证与日志归档）
- [x] 测试入口（目录推理、PNG 导出、耗时/显存/metrics JSON）
- [x] 创建 outputs 子目录与占位符（ckpts/visuals/metrics）
- [x] 创建模型验证脚本（validate_model_shapes.py）

### 变更日志

#### 【2025-10-09】架构迁移至新版（Cross-Attention 融合）

**重大变更：从旧版架构迁移至新版架构**

按照 `docs/Vrt+spike 视频去模糊 Baseline 开发指导.md` 第14章"迁移改造步骤"严格执行：

1. **删除旧版融合模块**
   - 删除 `src/models/fusion/concat_1x1_pre_tmsa.py`
   - 移除所有 Concat→1×1 融合相关代码

2. **新增核心模块**
   - **`src/models/spike_temporal_sa.py`**：Spike 时间维 Self-Attention
     - `TemporalSelfAttentionBlock`：在每个空间位置 (h,w) 上沿时间维 T 做 Self-Attention
     - `SpikeTemporalSA`：多尺度版本，为每个尺度创建独立的 Block
     - 超参：`heads=4`（固定），`dropout=0`（固定），每尺度 1 个 Block（固定）
   
   - **`src/models/fusion/cross_attn_temporal.py`**：时间维 Cross-Attention 融合
     - `TemporalCrossAttnFuse`：Q 来自 RGB (Fr)，K/V 来自 Spike (Fs')
     - `MultiScaleTemporalCrossAttnFuse`：多尺度版本
     - 超参：`heads=4`（固定），`dropout=0`（固定）

3. **重构集成代码**
   - **`src/models/integrate_vrt.py`**：完全重写以支持新架构
     - **新流程**：
       1. Spike → SpikeEncoder3D → Fs_1..7
       2. Spike → TemporalSA → Fs'_1..7（时间维自注意力）
       3. RGB → VRT Stage → Fr_i（每个 Stage 完成后）
       4. CrossAttn 融合：Ff_i = CrossAttn(Q=Fr_i, K/V=Fs'_i)
       5. Ff_i 继续后续流程
     - **关键变化**：融合从"TMSA 之前"改为"TMSA 之后"
     - 构造函数新增参数：`tsa_heads=4`, `fuse_heads=4`

4. **配置文件更新**
   - **`configs/deblur/vrt_spike_baseline.yaml`**：
     - 新增 `MODEL.LAYERS: 7`（尺度数）
     - 新增 `MODEL.SPIKE_TSA.HEADS: 4`
     - 修改 `MODEL.FUSE.TYPE: TemporalCrossAttn`
     - 新增 `MODEL.FUSE.HEADS: 4`

5. **训练脚本更新**
   - **`src/train.py`**：
     - 读取配置中的 `tsa_heads` 和 `fuse_heads`
     - 传入 `VRTWithSpike` 构造函数

6. **验证工具**
   - **`scripts/validate_model_shapes.py`**：
     - 验证模型前向传播
     - 检查各尺度形状、通道数、时间维对齐
     - 输出 Spike 编码器、TemporalSA、融合各阶段的特征形状

**架构对比总结**：

| 对比维度 | 旧版 | 新版 |
|---------|------|------|
| 融合时机 | TMSA **之前** | TMSA **之后** |
| 融合维度 | 通道维（Concat） | 时间维（Attention） |
| Spike 时序建模 | ❌ 无 | ✅ 有（Temporal Self-Attention） |
| RGB 时序建模 | 融合后才做 | 融合前已完成（VRT TMSA） |
| 信息流向 | 对称融合 | RGB 为主（Query），Spike 辅助（Key/Value） |
| 融合模块 | `Concat1x1PreTMSA` | `TemporalCrossAttnFuse` + `SpikeTemporalSA` |

---

#### 【初始实现】旧版架构（已废弃）

- 初始化项目结构：`configs/`、`scripts/`、`src/`、`third_party/`、进度文档
- 重构配置：`configs/deblur/vrt_spike_baseline.yaml`（大写字段对齐实施指导）
- 新增脚本：`scripts/prepare_data.py`、`scripts/train.py`、`scripts/test.py`
- 新增数据模块骨架：`src/data/datasets/{spike_deblur_dataset.py, voxelizer.py}`、`src/data/collate_fns.py`
- 新增模型骨架：`src/models/{spike_encoder3d.py, integrate_vrt.py}`、~~`src/models/fusion/concat_1x1_pre_tmsa.py`~~（已删除）
- 完成实现：`SpikeEncoder3D`（多尺度，新增 temporal_strides 仅时间维下采样能力）；默认对齐当前 VRT（Stage1..7 不改变时间维 D），故设置 strides=1/1/1/1/1/1 以保证各尺度 `T_i = T`。
- ~~完成与 VRT 集成：`src/models/integrate_vrt.py` 在 `Stage1..7` 前逐尺度 Concat+1x1x1 融合~~（已废弃）
- 创建输出目录：`outputs/{ckpts,logs,metrics,visuals}`。
- 补全依赖：`requirements.txt` 对齐实施指导
- 对齐 VRT 配置：训练脚本读取 `MODEL.VRT_CFG`（若存在）以同步 `img_size/window_size`；将 `MODEL.CHANNELS_PER_SCALE` 传入 `VRTWithSpike` 逐尺度对齐。
- Warmup 调度：新增 `TRAIN.SCHED.WARMUP_STEPS`，实现 `LinearLR -> CosineAnnealingLR` 串联（为 0 时退化为纯 Cosine）。
- 验证指标：补充 `SSIM(Y)`，写入 `outputs/metrics/val_step_*.json` 与 TensorBoard（`val/ssim_y`）。
- 对齐元信息：`SpikeDeblurDataset` 从 `outputs/logs/align_x4k1000fps.txt` 注入 `meta.t0/t1`，帧号解析与 vendor 保持一致（`parse_frame_index_from_name`）。
- Collate 日志：`safe_spike_deblur_collate` 记录丢弃样本形状至 `outputs/logs/collate_drop.txt`。
- ~~融合与通道：`VRTWithSpike` 支持 `channels_per_scale` 配置，逐尺度 `Concat→1×1×1` 保持与 VRT 通道一致~~（已废弃）。
- 启动脚本：新增 `scripts/setup_env.{sh,ps1}` 与 `scripts/launch_train.{sh,ps1}`，对齐实施指导。

### 当前说明
- **架构状态**：已完成迁移至新版架构（Cross-Attention 融合），严格遵守开发指导文档要求
- **纯 VRT（无 Spike）对照实验**：按计划暂不开发，后续单独进行
- **数据标准化统计（train-only）**：当前仅有 `val`，待训练集下载完成后收敛到 train-only 再回写
- **VGG 版本/层**：保持现状实现（不改）
- **待验证项**：运行 `scripts/validate_model_shapes.py` 验证模型前向传播

### 备注
- 暂未引入 VRT/SpikeCV 子模块，仅保留 `third_party/` 占位，后续按需接入
- 配置中 `channels_per_scale` 需与 VRT 编码通道对齐；当前 VRT 第 1–7 阶段 `embed_dims` 恒为 120，故 SpikeEncoder3D 通道设为对应尺度 120
- 新架构中，融合点在各 Stage **完成后**（包含 TMSA），而非之前版本的"TMSA 入口前"
- 注意力头数 `heads=4` 已固定在代码与配置中，符合开发指导要求
- 每尺度 Temporal Self-Attention Block 数量固定为 1，dropout 固定为 0
- 若后续更换 VRT 变体导致时间维在编码端缩减（例如 5→3→2→1），仅需在构建 `SpikeEncoder3D` 时传入 `temporal_strides` 以对齐各尺度 `T_i`

### 下一步工作
1. 运行 `scripts/validate_model_shapes.py` 验证模型构建与前向传播
2. 确保 VRT 子模块正确引入（`third_party/VRT`）
3. 准备数据集并运行 `scripts/prepare_data.py`
4. 启动训练并监控指标（PSNR/LPIPS/SSIM）
5. 对照"无 Spike"VRT 验证性能提升


