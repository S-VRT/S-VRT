# S-VRT: Spike 增强视频恢复 Transformer

[English](README.md) | [简体中文](README_zh-CN.md)

S-VRT 是一个面向 RGB 帧与 Spike 相机流的视频去模糊/视频恢复研究代码库。项目基于 VRT/KAIR 训练栈，扩展了 RGB+Spike dataset、Spike 表征、结构化 fusion operator，以及基于 SCFlow 的 Spike 光流路径。

README 只保留项目入口和快速导航；RGB+Spike 与 SCFlow 的细节请看 [`docs/`](docs/)。

## 主要特性

- **RGB+Spike 视频恢复**：使用 RGB 图像和 Spike 相机数据训练 VRT 风格 restoration 模型。
- **两类输入契约**：兼容旧的 `concat` 输入，也支持结构化 `dual+fusion` 输入。
- **Spike 表征**：支持 TFP voxel bins 和 centered raw Spike windows。
- **Fusion operators**：支持 gated、attention、PASE residual、dual-scale temporal Mamba 等变体。
- **光流选择**：可使用 RGB SpyNet，也可使用 Spike SCFlow 和 `encoding25` artifact。
- **SCFlow 子帧策略**：collapsed early fusion 下支持 `mean_windows` 和 `compose_subframes`。
- **训练工具**：单机 DDP 启动脚本、数据预处理脚本、TensorBoard/W&B/SwanLab/Logfire 接入和 forward-debug 脚本。

## 仓库状态

当前 RGB+Spike 配置位于 [`options/`](options/)，不再以 `options/vrt/` 作为主入口。

常用配置：

| 配置 | 用途 |
|------|------|
| [`options/006_train_vrt_videodeblurring_gopro_rgbspike.json`](options/006_train_vrt_videodeblurring_gopro_rgbspike.json) | TFP4 concat + SpyNet baseline |
| [`options/gopro_rgbspike_server.json`](options/gopro_rgbspike_server.json) | 服务器 RGB+Spike dual+fusion + SCFlow 主配置 |
| [`options/gopro_rgbspike_server_debug.json`](options/gopro_rgbspike_server_debug.json) | Debug 尺寸服务器配置 |
| [`options/gopro_rgbspike_server_pase_residual.json`](options/gopro_rgbspike_server_pase_residual.json) | PASE residual fusion 变体 |
| [`options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json`](options/gopro_rgbspike_server_dual_scale_temporal_mamba_raw_window.json) | Raw-window21 + dual-scale temporal Mamba 变体 |

## 环境

项目通常运行在已准备好的 Python/PyTorch 环境中。在仓库根目录：

```bash
cd /root/projects/S-VRT
uv pip list >/dev/null
```

如果不用 `uv`，请在当前 Python 环境中安装项目依赖。核心运行依赖包括 PyTorch、OpenCV、NumPy、einops、timm、tensorboard、wandb、swanlab 和 pytest。

DCNv4 等可选 CUDA 扩展只在所选配置需要时构建。

## 数据目录

GoPro + Spike 推荐结构：

```text
gopro_spike/
├── GOPRO_Large/
│   ├── train_GT/
│   ├── train_GT_blurred/
│   ├── test_GT/
│   └── test_GT_blurred/
└── GOPRO_Large_spike_seq/
    ├── train/
    │   └── <clip>/
    │       └── spike/
    │           └── *.dat
    ├── test/
    └── config.yaml
```

服务器配置当前指向：

```text
/root/autodl-tmp/datasets/gopro_spike/GOPRO_Large
/root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq
```

## Spike Artifact 预处理

TFP artifact：

```bash
uv run python scripts/data_preparation/spike/prepare_spike_tfp.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --num-bins 4 \
  --half-win-length 20 \
  --workers 32
```

Raw-window artifact：

```bash
uv run python scripts/data_preparation/spike/prepare_spike_raw_window.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --spike-h 360 \
  --spike-w 640 \
  --window-length 21 \
  --workers 32
```

SCFlow `encoding25` artifact：

```bash
uv run python scripts/data_preparation/spike_flow/prepare_scflow_encoding25.py \
  --spike-root /root/autodl-tmp/datasets/gopro_spike/GOPRO_Large_spike_seq/train \
  --dt 10 \
  --subframes 4 \
  --spike-h 360 \
  --spike-w 640 \
  --num-workers 16
```

Raw-window21 配置需要使用 `--subframes 21` 生成对应 SCFlow artifact。

## 快速开始

Concat baseline 的 Dataset smoke test：

```bash
uv run python - <<'PY'
from utils import utils_option as option
from data.select_dataset import define_Dataset

opt = option.parse("options/006_train_vrt_videodeblurring_gopro_rgbspike.json", is_train=True)
ds = define_Dataset(opt["datasets"]["train"])
sample = ds[0]
print(sample["L"].shape)
print(sample["H"].shape)
PY
```

运行 debug 训练：

```bash
bash launch_train.sh 1 options/gopro_rgbspike_server_debug.json --foreground
```

运行多 GPU 后台训练：

```bash
bash launch_train.sh 4 options/gopro_rgbspike_server.json --detach
```

如果平台已经注入 DDP 环境变量，不要再套 `torchrun`：

```bash
uv run python -u main_train_vrt.py --opt options/gopro_rgbspike_server.json
```

## 测试

常用契约和脚本测试：

```bash
uv run pytest tests/test_forward_debug_batch_script.py -v
uv run pytest tests/models/test_optical_flow_scflow_contract.py tests/models/test_optical_flow_scflow_integration.py -v
uv run pytest tests/models/test_fusion_early_adapter.py tests/models/test_vrt_fusion_integration.py -v
```

服务器配置 smoke test：

```bash
uv run pytest tests/e2e/test_gopro_rgbspike_server_e2e.py -v
```

## 文档

| 文档 | 内容 |
|------|------|
| [`docs/QUICKSTART_RGB_SPIKE.md`](docs/QUICKSTART_RGB_SPIKE.md) | 当前 RGB+Spike 操作指南 |
| [`docs/RGB_SPIKE_DESIGN.md`](docs/RGB_SPIKE_DESIGN.md) | 数据/模型契约与设计理由 |
| [`docs/RGB_SPIKE_IMPLEMENTATION.md`](docs/RGB_SPIKE_IMPLEMENTATION.md) | 实现模块、脚本、配置和验证入口 |
| [`docs/SCFLOW_SUBFRAME_FUSION.md`](docs/SCFLOW_SUBFRAME_FUSION.md) | SCFlow 子帧集成和 collapse policy |
| [`docs/SGP_MODULE.md`](docs/SGP_MODULE.md) | SGP 模块说明 |
| [`docs/SPIKE_ENCODER_DESIGN.md`](docs/SPIKE_ENCODER_DESIGN.md) | Spike encoder 设计说明 |

## 项目结构

```text
S-VRT/
├── main_train_vrt.py
├── main_test_vrt.py
├── launch_train.sh
├── launch_test.sh
├── data/
│   ├── dataset_video_train_rgbspike.py
│   ├── dataset_video_test.py
│   └── spike_recc/
├── models/
│   ├── architectures/vrt/
│   ├── fusion/
│   └── optical_flow/
├── options/
├── scripts/
│   ├── data_preparation/
│   └── analysis/
├── tests/
└── docs/
```

## 引用

如果使用本仓库，请引用原始 VRT 工作，以及实验中使用到的 Spike/SCFlow 相关组件：

```bibtex
@article{liang2022vrt,
  title={VRT: A Video Restoration Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Fan, Yuchen and Zhang, Kai and Ranjan, Rakesh and Li, Yawei and Timofte, Radu and Van Gool, Luc},
  journal={arXiv preprint arXiv:2201.12288},
  year={2022}
}
```

## 致谢

- [VRT](https://github.com/JingyunLiang/VRT)
- [KAIR](https://github.com/cszn/KAIR)
- 本仓库 vendored 或引用的 SpikeCV 与 optical-flow 组件

## 许可证

本仓库基于 KAIR/VRT 组件构建。重新分发或商业使用前，请检查仓库许可证和上游组件许可证。
