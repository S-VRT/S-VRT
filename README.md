# Spike-VRT: Video Restoration with Spike Camera Data

基于VRT（Video Restoration Transformer）架构的视频恢复项目，专门用于处理Spike相机数据的视频去模糊任务。

## 项目简介

Spike-VRT (S-VRT) 是一个基于VRT架构的视频恢复框架，通过融合RGB图像和Spike相机数据来提升视频去模糊性能。本项目扩展了原始VRT架构，支持多模态输入（RGB + Spike），特别适用于处理运动模糊场景。

### 主要特性

- **多模态融合**：同时使用RGB图像和Spike相机数据进行视频恢复
- **视频去模糊**：专门针对运动模糊场景优化
- **Spike数据支持**：内置Spike数据加载和处理工具
- **分布式训练**：支持单GPU和多GPU分布式训练
- **灵活配置**：支持多种数据集和训练配置
- **DCNv4支持**：可选使用DCNv4代替DCNv2，提升性能并保持完全兼容

## 环境要求

### 依赖安装

```bash
pip install -r requirement.txt
```

主要依赖包括：
- PyTorch
- opencv-python
- scikit-image
- pillow
- torchvision
- timm
- einops
- tensorboard
- wandb
- swanlab
- logfire

## 数据集准备

### GoPro + Spike数据集

本项目使用GoPro数据集配合Spike相机数据。数据集结构如下：

```
gopro_spike/
├── GOPRO_Large/
│   ├── train_GT/              # 训练集清晰图像
│   ├── train_GT_blurred/     # 训练集模糊图像
│   ├── test_GT/               # 测试集清晰图像
│   └── test_GT_blurred/       # 测试集模糊图像
└── GOPRO_Large_spike_seq/
    ├── train/                 # 训练集Spike数据
    │   └── [sequence_name]/
    │       └── spike/
    │           └── *.dat     # Spike数据文件
    ├── test/                  # 测试集Spike数据
    └── config.yaml            # Spike相机配置
```

### 自动数据准备

使用提供的脚本自动准备数据集：

```bash
# 使用默认路径
python scripts/data_preparation/prepare_gopro_spike_dataset.py

# 指定路径
python scripts/data_preparation/prepare_gopro_spike_dataset.py \
    --gopro_root /path/to/GOPRO_Large \
    --spike_root /path/to/GOPRO_Large_spike_seq
```

### DCNv4 支持（可选）

项目支持 DCNv4 作为可选的可变形卷积实现，提供更好的性能和速度。DCNv4 需要 CUDA 支持。

#### 构建 DCNv4

```bash
# 进入 DCNv4 目录
cd models/op/dcnv4

# 构建 CUDA 扩展
./build.sh

# 或手动构建
python setup.py build_ext --inplace
```

#### 使用 DCNv4

在配置文件中设置：
```json
{
  "netG": {
    "dcn_type": "DCNv4"
  }
}
```

默认配置使用 DCNv2：
```json
{
  "netG": {
    "dcn_type": "DCNv2"
  }
}
```

## 训练

### 快速开始

使用提供的启动脚本进行训练：

```bash
# 单GPU训练
./launch_train.sh 1

# 多GPU训练（4个GPU）
./launch_train.sh 4

# 指定配置文件
./launch_train.sh 4 options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json

# 训练前自动准备数据
./launch_train.sh 1 --prepare-data
```

### 手动训练

#### 单GPU训练

```bash
python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

#### 多GPU分布式训练

```bash
# 使用torchrun
torchrun --nproc_per_node=4 main_train_vrt.py \
    --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json

# 或使用环境变量（平台DDP）
python -u main_train_vrt.py \
    --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

### 配置文件说明

主要配置文件位于 `options/vrt/` 目录下：

- `006_train_vrt_videodeblurring_gopro_rgbspike.json` - GoPro + Spike视频去模糊训练配置

关键配置项：

```json
{
  "netG": {
    "in_chans": 7,  // 输入通道数：3 (RGB) + 4 (Spike TFP)
    "img_size": [6, 224, 224],
    "window_size": [6, 8, 8],
    "dcn_type": "DCNv2"  // 可选: "DCNv2" (默认) 或 "DCNv4"
  },
  "datasets": {
    "train": {
      "dataset_type": "VideoRecurrentTrainDatasetRGBSpike",
      "dataroot_gt": "path/to/train_GT",
      "dataroot_lq": "path/to/train_GT_blurred",
      "dataroot_spike": "path/to/spike/train",
      "spike_h": 250,
      "spike_w": 400,
      "spike_channels": 4
    }
  }
}
```

### DCNv4支持

项目支持可选使用DCNv4代替传统的DCNv2，通过在配置中设置 `"dcn_type": "DCNv4"` 来启用：

- **DCNv2** (默认): 传统可变形卷积，稳定可靠
- **DCNv4**: 最新可变形卷积版本，提供更好的性能和收敛速度

DCNv4与DCNv2完全兼容，无需修改其他配置项。DCNv4会自动编译安装，使用前请确保CUDA环境正确配置。

## 测试

### 快速使用 `launch_test.sh`

`launch_test.sh` 与 `launch_train.sh` 共用同一份 JSON 配置（默认 `options/vrt/gopro_rgbspike_local.json`），
脚本会读取其中的 `datasets.test`、`val` 等字段并生成一份运行期配置，然后调用
`python main_test_vrt.py --opt <runtime_config>`。测试所需的任务名、分块策略、保存开关等都应在 JSON 中配置。
如果你的数据根目录与 JSON 中不同，可继续使用 `--dataset-root`、`--gopro-root` 等参数覆盖，
脚本会在生成的运行期配置里注入新的路径，整个流程仍只需要维护一份 JSON。

```bash
# 查看所有选项
./launch_test.sh --help

# 使用默认 JSON 配置运行测试（单 GPU）
./launch_test.sh 1

# 指定另一份 JSON 配置
./launch_test.sh 1 options/vrt/gopro_rgbspike_ablation.json

# 指定另一份 JSON 并覆盖数据根目录
./launch_test.sh 1 options/vrt/gopro_rgbspike_ablation.json \
    --dataset-root /data/gopro_spike_custom
```

常用参数说明：
- `CONFIG_PATH`：可选位置参数，指向训练/测试 JSON；缺省为 `options/vrt/gopro_rgbspike_local.json`
- `GPU_COUNT / --gpus`: 指定使用的 GPU 数量或编号（当前脚本仅启动单进程推理）
- `--dataset-root` / `--gopro-root` / `--spike-root`: 快速覆盖数据根目录，脚本会在临时 JSON 中重写 `datasets.*.dataroot_*`
- `--prepare-data`, `--generate-lmdb`, `--force-prepare`: 复用训练脚本的数据准备能力

> 📌 需要修改任务名、分块大小、是否保存结果等行为时，请直接编辑 JSON
（例如 `task`/`val.task_name`、`val.num_frame_testing`、`val.size_patch_testing`,
`val.save_img` 等字段），脚本会自动读取这些配置并传递给 `main_test_vrt.py`。

### 直接调用 `main_test_vrt.py`

推荐方式是与训练一致，直接传入 JSON：

```bash
python main_test_vrt.py --opt options/vrt/gopro_rgbspike_local.json
```

若需要快速实验，也可以沿用旧的 CLI 方式手动指定参数：

```bash
python main_test_vrt.py \
    --task 006_VRT_videodeblurring_GoPro \
    --folder_lq testsets/GoPro/test_GT_blurred \
    --folder_gt testsets/GoPro/test_GT \
    --tile 40 192 192 \
    --tile_overlap 2 20 20 \
    --save_result
```

### 测试参数

- `--task`: 任务类型
- `--folder_lq`: 低质量（模糊）视频文件夹
- `--folder_gt`: 高质量（清晰）视频文件夹（可选，用于评估）
- `--tile`: 分块测试大小 `[temporal, height, width]`，`[0,0,0]` 表示不分块
- `--tile_overlap`: 分块重叠大小
- `--save_result`: 保存结果图像

## 项目结构

```
S-VRT/
├── main_train_vrt.py              # 训练主程序
├── main_test_vrt.py               # 测试主程序
├── launch_train.sh                # 训练启动脚本
├── models/                        # 模型定义
│   ├── network_vrt.py             # VRT网络架构
│   ├── model_vrt.py               # VRT模型封装
│   └── ...
├── options/                        # 配置文件
│   └── vrt/                       # VRT相关配置
├── data/                          # 数据加载
│   └── dataset_video_train_rgbspike.py  # RGB+Spike数据集
├── utils/                         # 工具函数
│   ├── spike_recc/            # Spike数据加载与重建 (Spike Reconstruction)
│   │   ├── spikecv_loader.py  # 基于SpikeCV的加载器
│   └── ...
├── scripts/                       # 脚本工具
│   └── data_preparation/          # 数据准备脚本
└── SpikeCV/                       # SpikeCV库（可选）
```

## Spike数据格式

本项目支持Spike相机的`.dat`格式数据：

- **格式**：二进制文件，包含时间序列的spike事件
- **分辨率**：默认250×400（可在配置文件中修改）
- **加载**：使用 `data/spike_recc/spikecv_loader.py` 中的工具加载

### Spike数据加载示例

```python
from data.spike_recc import SpikeStreamSimple

# 加载Spike数据
spike_stream = SpikeStreamSimple(
    filepath="path/to/spike.dat",
    spike_h=250,
    spike_w=400
)

# 获取spike矩阵
spike_matrix = spike_stream.get_spike_matrix(flipud=True)  # (T, H, W)
```

## 模型架构

Spike-VRT基于VRT架构，主要改进：

1. **多模态输入**：网络输入通道从3（RGB）扩展到4（RGB + Spike）
2. **Spike融合**：在Transformer架构中融合Spike时序信息
3. **视频去模糊**：针对运动模糊场景优化的损失函数和训练策略

## 日志和监控

训练过程支持 TensorBoard、WANDB、SwanLab 与 Logfire 日志记录：

- **TensorBoard**：`tensorboard --logdir experiments/[experiment_name]/tb_logger`
- **WANDB**：在配置文件中设置 `wandb_api_key` 与 `wandb_project`（或使用 `WANDB_API_KEY` 环境变量）
- **SwanLab**：在配置文件中设置 `swanlab_project`/`swanlab_api_key` 或通过 `SWANLAB_API_KEY` + `swanlab login` 完成授权，使用 `swanlab_mode: "offline"` 可在无网络环境记录。默认会在 `experiments/<task_name>/swanlab_run.id` 中缓存云端 run id，便于在恢复训练（例如从 10k 步 checkpoint 继续）时自动续写同一次 run；如需开启新 run，可删除该文件或将 `swanlab_auto_resume` 置为 `false`。
- **Logfire**：在配置文件的 `logging` 段设置 `use_logfire`、`logfire_token`、`logfire_project_name`、`logfire_service_name`、`logfire_environment`。启用后，训练文本日志以及 `train/*`、`test/*`、`time/*` 指标会并行发送到 Logfire；Logfire 异常不会中断训练。

## 常见问题

### 1. 内存不足

如果遇到内存不足问题，可以：
- 减小 `dataloader_batch_size`
- 减小 `gt_size`
- 使用分块测试（设置 `--tile` 参数）

### 2. Spike数据加载失败

检查：
- Spike数据路径是否正确
- `spike_h` 和 `spike_w` 是否与数据匹配
- `.dat` 文件是否完整

### 3. 分布式训练问题

- 确保使用 `torchrun` 或平台提供的DDP环境
- 检查 `WORLD_SIZE` 和 `RANK` 环境变量
- 使用 `launch_train.sh` 脚本可以自动处理

## 引用

如果使用本项目，请引用原始VRT论文：

```bibtex
@article{liang2022vrt,
title={VRT: A Video Restoration Transformer},
author={Liang, Jingyun and Cao, Jiezhang and Fan, Yuchen and Zhang, Kai and Ranjan, Rakesh and Li, Yawei and Timofte, Radu and Van Gool, Luc},
  journal={arXiv preprint arXiv:2201.12288},
year={2022}
}
```

## 许可证

本项目基于原始KAIR/VRT项目，请参考LICENSE文件。

## 致谢

- [VRT](https://github.com/JingyunLiang/VRT) - 原始VRT实现
- [KAIR](https://github.com/cszn/KAIR) - 训练框架基础
