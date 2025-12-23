
# 1. 目标与原则（Cursor 执行的“重构边界”）

## 1.1 你要得到什么

* 一个 **OpenMMLab 风格的算法库包**（像 `mmdet/` 那样），你的算法库建议命名为 `mmvrt/`（避免与 mmdet 冲突）。
* 根目录只保留：

  * `mmvrt/`（算法库本体）
  * `configs/`（实验配置）
  * `tools/`（训练/测试/推理/数据准备入口脚本）
  * 以及必要的 `README/requirements/pyproject` 等工程文件
* 训练/验证/测试**完全走 MMEngine Runner.from_cfg** 的配置驱动路径；你现在的自研 `core/engine/train` 将大幅收敛或删除，避免“重复造轮子”。([mmengine.readthedocs.io][2])

## 1.2 三条硬原则（Cursor 实施时必须遵守）

1. **Framework 能力不自研**：runner、hooks、logging、checkpoint、optim_wrapper、evaluator 的主体使用 MMEngine；你只写“算法库组件”。([mmengine.readthedocs.io][2])
2. **实验=配置**：所有变体（TFP/TDM、SGP、early/late fusion、窗口数 K、帧数 T 等）都通过 `configs/` 的继承覆盖来管理（像 MMDet 那样）。([MMDetection][1])
3. **数据=Dataset + Pipeline + DataPreprocessor**：Dataset 只负责索引与读样本；“TFP/TDM/裁剪/翻转/打包”等都在 transforms；归一化/颜色/模态拼接等尽量下沉到 data_preprocessor（这点与 MMagic 的迁移原则一致）。([MMDetection][3])

---

# 2. 最终目录结构（对标 mmdet/ 的“包内分域”）

> 你给的旧结构会变成下面这种 **“一个包 + configs + tools”** 的标准形态（MMDet 就是这种组织方式）。([MMDetection][1])

## 2.1 根目录

* `mmvrt/`：算法库包（对标 `mmdet/`）
* `configs/`：实验配置（按任务与变体组织）
* `tools/`：训练/测试/推理/数据准备入口（脚本入口层）
* `tests/`：单元测试与集成测试（可选但强烈建议）
* `README.md` / `pyproject.toml`（或 `setup.cfg`）等工程文件

## 2.2 包内（mmvrt/）分域（推荐与 MMDet 对齐）

* `mmvrt/apis/`：高层推理 API（可选；更偏“产品化”）
* `mmvrt/structures/`：数据结构（DataSample、PackInputs 的契约）
* `mmvrt/datasets/`：Dataset 与 Transforms（pipeline）
* `mmvrt/models/`：模型组件（restorer/backbone/loss/…）
* `mmvrt/engine/`：训练时自定义 hooks / loops（仅保留必要扩展）
* `mmvrt/evaluation/`：metric & evaluator（PSNR/SSIM/视频指标）
* `mmvrt/visualization/`：可视化与保存（视频拼接、对比图）
* `mmvrt/utils/`：通用工具（与任务无关、纯函数/薄封装）
* `mmvrt/registry.py`：本库 registry 与 default_scope 入口（关键）
* `mmvrt/version.py` / `mmvrt/__init__.py`

> MMDet 的“7 大部分”即 apis/structures/datasets/models/engine/evaluation/visualization，你这里完全对齐，只是任务从 detection 换成 restoration。([MMDetection][1])

---

# 3. Registry 与 Default Scope（让“配置驱动组装”真正成立）

**目标**：让配置里的 `type=...` 可以像 MMDet 一样构建你的组件；并避免注册名冲突。

## 3.1 default_scope

* 设定 default_scope 为 `"mmvrt"`（对标 OpenMMLab 下游库习惯）。
* 配置中可以少写 registry scope，体验与 MMDet 接近。

## 3.2 你需要哪些 registries（最小集合）

* MODELS：restorer/backbone/head/loss/data_preprocessor 统归这里（像很多 OpenMMLab 库做法）
* DATASETS：各类 dataset
* TRANSFORMS：pipeline transforms（数据增强与 spike 表示构建）
* METRICS：PSNR/SSIM 等
* HOOKS：必要时自定义 hook（例如可视化保存、额外日志、特殊 EMA）

> MMEngine 的 Runner / BaseModel 设计就是为了“从 cfg 构建组件并训练”，你的库要做的是把组件注册好并遵循接口。([mmengine.readthedocs.io][2])

---

# 4. Model 设计（解决你提出的 1&2：单文件膨胀 + 变体体系）

你现在的问题，本质是：**“VRT=网络结构+训练逻辑+推理逻辑+loss glue 全耦合”**。对标 OpenMMLab，必须拆成“胶水模型 + 可替换组件”。

## 4.1 顶层模型类型：Restorer（对标 MMDet 的 Detector 概念）

在 `mmvrt/models/restorers/` 下定义：

* `BaseRestorer`：定义你任务的统一 forward 协议与 data_sample 约定
* `VRTDeblurRestorer`：**胶水层**

  * 持有：backbone / (可选 neck) / head / loss / data_preprocessor
  * 负责：train 计算 loss、val/test 产出预测与可评估输出

> MMEngine 要求模型实现统一的 forward，并区分 `mode='loss'/'predict'/'tensor'` 的语义；这是你“完全对标”的关键接口点。([mmengine.readthedocs.io][4])

## 4.2 组件化拆分（把 `network_vrt.py` 拆开）

推荐 `mmvrt/models/` 结构：

* `backbones/`

  * `vrt_backbone.py`：标准 VRT 主干
  * `sgp_vrt_backbone.py`：SGP 变体（只改 block，不动 restorer 语义）
* `layers/`

  * attention/block/sgp 等更小颗粒部件（让未来改结构不改大文件）
* `motion/`

  * `spynet.py` 及 flow/align 相关（从“模型主文件”剥离）
* `heads/`

  * `deblur_head.py`：把“输出重建层”独立出来（未来可替换不同 head）
* `losses/`

  * pixel/perceptual/charbonnier 等
* `data_preprocessors/`

  * `rgb_spike_preprocessor.py`：做模态拼接、归一化、padding/stack 等（见第 6 节）

## 4.3 “抽象类-各种变体”的正确落点（避免继承爆炸）

对标 OpenMMLab 的推荐做法：**变体优先通过“可替换组件”实现，而不是靠大量子类继承。**

* **结构变体（SGP / 不同注意力 / 不同对齐）**：放在 `backbones/` 或 `layers/`，配置切换 `type` 即可
* **模态变体（early-fusion/late-fusion、K 分桶数、不同 spike 表示）**：优先在 `data_preprocessor` + `transforms` 参数化
* **训练语义变体（额外辅助 loss、多阶段训练、特殊推理）**：才考虑新增 `restorer` 子类（少而精）

这样你未来做变体，95% 只需要：

* 新增一个组件类文件（backbone/layer/transform）
* 新增一个 config 覆盖文件
  而不是复制粘贴一个巨大 model.py。

> 这种“模块化组合”正是 MMDet 在 repo 里强调的 major features（Modular Design）。([GitHub][5])

---

# 5. Dataset 与 Pipeline 设计（解决你提出的 3：dataset_* 文件职责混乱）

你现在的 `dataset_video_train_rgbspike.py` 这类“train 专用 dataset 文件”应当消失，换成 MMDet 典型的 **Dataset + Pipeline** 分离：Dataset 产出原始样本 dict，Pipeline 逐步加工。([MMDetection][3])

## 5.1 Dataset 层（mmvrt/datasets/datasets/）

只做三件事：

1. 管理索引（视频片段、帧号、对应 spike 文件、GT）
2. 读取必要的原始数据（或读取路径交给 Load transform）
3. 输出统一字段的 sample dict（不做增强、不做 TFP/TDM）

建议的 Dataset 划分：

* `BaseVideoDataset`：视频片段抽象（clip_len、stride、temporal sampling）
* `GoProRGBSpikeDataset`：你当前任务的数据组织实现（文件结构与标注协议）
* `RGBSpikeTestDataset`：测试/benchmark 的变体（如果只在采样策略上不同，尽量用同一个 Dataset + 不同 config）

## 5.2 Pipeline/Transforms 层（mmvrt/datasets/transforms/）

对标 MMDet 的 pipeline 思想：一串 transform，每个 transform 输入 dict 输出 dict，最终 pack 成 DataSample。([MMDetection][3])

你需要的 transforms 类别（设计分组）：

### A. I/O transforms（加载）

* LoadRGBFrames（支持按 clip 读帧）
* LoadSpikeRaw（读 spike 流 / spike tensor / spike 文件）
* LoadGT（GT 帧或清晰帧）

### B. Spike 表示 transforms（你最关键）

* SpikeToTFP（TFP 重建/积分窗口参数化）
* SpikeToTDM / SpikeToTLA（如果保留对比）
* TemporalBinning（K 分桶策略：均匀/中心对齐/滑窗）

> 这里“参数全来自 config”：窗口长度、对齐方式、K、是否中心对齐 RGB 时间戳等。

### C. Augment transforms（增强与裁剪）

* RandomCrop（对齐 RGB+Spike+GT 一起裁）
* RandomFlip / TemporalReverse（如果允许）
* Normalize/ColorSpace：**尽量不在这里做**，下沉到 data_preprocessor（对齐 MMagic 建议）([mmagic.readthedocs.io][6])

### D. Formatting transforms（打包）

* PackRestorationInputs：把 dict 封装成 DataSample，包含 metainfo（视频名、帧号、时间戳、尺度等）

## 5.3 Train/Test 的差异放哪里？

* **不要**用两个 dataset 文件区分 train/test
* **要**用两个 dataloader 配置项（train_dataloader / val_dataloader / test_dataloader）以及不同 pipeline 列表区分。
  这就是 MMEngine Runner 的标准输入形态。([mmengine.readthedocs.io][2])

---

# 6. DataPreprocessor 设计（解决“模态拼接、归一化、对齐”的归属）

你现在很多逻辑可能散落在 dataset 或 model forward 里。对标 OpenMMLab（尤其 MMagic 的迁移说明）：**Normalization/Color space 等从 transforms 移到 data_preprocessor**，在模型前统一处理。([mmagic.readthedocs.io][6])

## 6.1 `RGBSpikeDataPreprocessor`（mmvrt/models/data_preprocessors/）

职责建议明确为：

* 把 pipeline 输出的 RGB / Spike / GT 变成模型所需的 tensor 组织形式
* 统一做：

  * normalization（RGB 的 mean/std）
  * spike 的尺度归一（如果需要）
  * batch 维度堆叠与 padding（不同尺寸视频/裁剪边界）
  * **early-fusion / late-fusion 的输入组织**（关键！）

## 6.2 Early vs Late fusion 的落点

* **Early-fusion**：preprocessor 直接输出 `inputs = concat([rgb, spike_repr], dim=channel)` 给 backbone
* **Late-fusion**：preprocessor 分别输出 `inputs_rgb` 与 `inputs_spike`，由 restorer 组装成双流 backbone 或 backbone+fusion neck

这样，你未来所有 fusion 变体主要改：

* preprocessor 的输出字段与组织
* 或新增一个 `FusionNeck` 组件
  而不是改整个 VRT 主干文件。

---

# 7. Structures（统一“样本契约”，避免字段散乱）

MMDet 强调 structures 的作用：作为组件间接口（DetDataSample 等）。你这里也需要一个 restoration 版本，避免“dict 乱飞”。([MMDetection][7])

## 7.1 定义 RestorationDataSample（mmvrt/structures/）

建议字段设计（设计级，不写代码）：

* `.inputs`：模型输入（可能是 RGB+Spike 拼接，或 dict）
* `.gt`：ground truth
* `.pred`：预测输出（推理后）
* `.metainfo`：视频名、帧号、时间戳、裁剪位置、尺度、数据源等

## 7.2 Pack transform 只负责“组装”

最后一个 transform（PackRestorationInputs）把 dict → RestorationDataSample，确保后续 restorer/metric/visualization 都只认 DataSample。

---

# 8. Engine / Evaluation / Visualization（你现在 metrics/ema/hooks 的归位）

## 8.1 Engine：只保留“任务特有”hook

MMEngine 已有默认 hooks（logger、checkpoint、param_scheduler 等），你只保留必要扩展：

* 可视化保存 hook（定期保存对比视频/帧拼图）
* 自定义 EMA（如果你的 EMA 实现与默认不同）
* 训练时 profile hook（可选）

> 重点：不要再维护两套 runner。Runner 交给 MMEngine。([mmengine.readthedocs.io][2])

## 8.2 Evaluation：PSNR/SSIM 与视频指标

* `mmvrt/evaluation/metrics/`：PSNR、SSIM、（可选）LPIPS、（可选）时序一致性指标
* `mmvrt/evaluation/evaluator/`：把 metrics 组合起来供 Runner 使用

## 8.3 Visualization：对比输出标准化

* 统一输出目录结构（work_dir 下）
* 标准产物：

  * 单帧对比图（input blur / pred / gt）
  * 视频对比（按 clip 拼接）
  * spike 表示可视化（TFP/TDM 图）

---

# 9. Configs 组织（让“变体管理”像 MMDet 一样干净）

MMDet 的 configs 组织核心是：***base* 复用 + 变体覆写**。([MMDetection][1])

## 9.1 建议 configs 目录

* `configs/vrt_deblur/`

  * `_base_/`

    * `datasets/`：gopro_rgbspike.py（train/val/test dataloader + pipeline）
    * `runtime/`：default_runtime.py（log、ckpt、env、seed）
    * `schedules/`：iter-based schedule（lr、warmup、total iters）
    * `models/`：vrt_rgb_only.py、vrt_rgb_spike_base.py（模型大骨架）
  * `vrt_rgb_only.py`
  * `vrt_rgb_spike_tfp_k4.py`
  * `vrt_rgb_spike_tdm_k8.py`
  * `vrt_sgp_rgb_spike_tfp_k4.py`
  * `ablation/`：专门放消融

## 9.2 你要把哪些东西“配置化”（务必列给 Cursor）

* 模型：

  * backbone type（VRT / SGP-VRT）
  * spike 融合方式（early/late）
  * clip_len、输入通道组织策略
* 数据：

  * spike 表示（TFP/TDM/TLA）
  * K 分桶数、窗口长度、对齐策略
  * crop size、augmentation
* 训练：

  * optim_wrapper（adamw 等）
  * lr schedule、iters、val_interval
  * default_hooks（logger、checkpoint）
* 评估：

  * metrics 列表与计算口径（RGB/ Y channel 等）

---

# 10. Tools 与 Data Prepare（解决你提出的 4：prepare_data.py 的位置与职责）

对标 MMDet：**工具脚本在 tools/**，并且常有 dataset 浏览、转换、评估等工具。([MMDetection][8])

## 10.1 tools/ 应包含的脚本类别

* `tools/train.py`：读取 config → Runner.from_cfg → train（标准入口）([mmengine.readthedocs.io][2])
* `tools/test.py`：同 config → test
* `tools/infer.py`：单视频/文件夹推理（可选）
* `tools/data/prepare_gopro_rgbspike.py`：数据准备（从原始到可训练格式）
* `tools/misc/browse_dataset.py`：可视化 dataset 与 pipeline 输出（对标 MMDet 的 browse_dataset 思路）([MMDetection][8])

## 10.2 prepare_data 的“合理职责边界”

* 负责：

  * 扫描原始数据目录
  * 生成索引（json/csv）
  * （可选）转 LMDB / shards
  * 数据一致性校验（RGB 与 spike 时间戳/帧号匹配）
* 不负责：

  * 训练时的增强/TFP/TDM（那些属于 transforms/pipeline）

---

# 11. 迁移实施路线图（Cursor 按阶段落地，不会炸）

> 这是给 Cursor 的“分阶段任务单”，每一步都能独立验收。

## Phase A：搭骨架（不动算法）

1. 新建 `mmvrt/` 包与分域目录（apis/structures/datasets/models/engine/evaluation/visualization/utils）
2. 建立 `mmvrt/registry.py` 与 default_scope 约定
3. 建立 `configs/` 与 `tools/` 基础入口（先空跑，确保 Runner 能启动）([mmengine.readthedocs.io][2])

**验收**：能用一个最小 config 启动 Runner（哪怕模型是 dummy）。

## Phase B：迁移数据链路（先跑通 dataloader）

1. 把 spike_loader 的“文件读取/解码”迁移到 transforms 的 I/O 层
2. 把 `dataset_video_*` 逻辑迁移成 Dataset + Pipeline
3. 增加 PackRestorationInputs + RestorationDataSample

**验收**：`tools/misc/browse_dataset.py` 能看到 pipeline 输出，字段稳定（不依赖训练）。([MMDetection][3])

## Phase C：迁移模型为 Restorer + Backbone（不改数学逻辑）

1. 把当前 VRT 大文件拆成：

   * restorer（胶水）
   * backbone（结构）
   * motion/layers（子模块）
   * losses
2. 定义 data_preprocessor，并把 normalization/拼接/stack 迁移进去（对齐 MMagic 思路）([mmagic.readthedocs.io][6])

**验收**：在不追求指标的情况下，forward(mode=loss/predict) 能跑通，并可保存 ckpt。([mmengine.readthedocs.io][4])

## Phase D：补齐 evaluation/visualization 与变体 configs

1. 实现 PSNR/SSIM metrics 与 evaluator 接线
2. 把所有变体做成 configs 覆盖（TFP/TDM、K、SGP、fusion）
3. 添加可视化 hook（定期保存对比）

**验收**：不同 config 之间只改配置文件即可切换实验；work_dir 产物一致。

---

# 12. 你现在旧目录到新目录的“映射表”（给 Cursor 直接搬运）

* `models/vrt/network_vrt.py` → `mmvrt/models/backbones/vrt_backbone.py` + `mmvrt/models/layers/*` + `mmvrt/models/motion/*`
* `models/vrt/sgp_vrt.py` → `mmvrt/models/backbones/sgp_vrt_backbone.py`（只存差异 block）
* `models/vrt/model.py` → `mmvrt/models/restorers/vrt_deblur_restorer.py`（胶水层）
* `models/losses/*` → `mmvrt/models/losses/*`
* `data/datasets/*` + `dataset_video_*` → `mmvrt/datasets/datasets/*` + `mmvrt/datasets/transforms/*`
* `utils/spike_loader.py` → `mmvrt/datasets/transforms/loading.py` 或 `mmvrt/datasets/io/spike.py`
* `engine/metrics.py` → `mmvrt/evaluation/metrics/*`
* `engine/hooks.py` → `mmvrt/engine/hooks/*`（仅保留必要）
* `tools/prepare_data.py` → `tools/data/prepare_*.py`
* `core/*`（builder/registry/runtime）→ **大部分删除或并入 `mmvrt/registry.py` 的薄封装**；runner 交给 MMEngine ([mmengine.readthedocs.io][2])

---

# 13. 最终你会拥有的“工程能力”（对标达成标准）

当这套重构完成，你应该具备这些“MMDet 同级体验”：

* 任何变体只需新增一个组件文件 + 一个 config 文件（不改主训练脚本）
* `tools/train.py xxx.py` 一键训练；`tools/test.py` 一键评估（同一份 cfg 驱动）([mmengine.readthedocs.io][2])
* 数据链路可视化、可调试（pipeline 每一步清晰）([MMDetection][3])
* 模型 forward 语义与 MMEngine 标准一致（loss/predict/tensor）([mmengine.readthedocs.io][9])
* normalization/拼接/模态组织在 data_preprocessor，dataset 与 transforms 职责干净（符合 MMagic 的复原任务范式）([mmagic.readthedocs.io][6])

---

