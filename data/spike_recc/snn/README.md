# SNN (脉冲神经网络) 使用说明

## 目录

1. [简介](#简介)
2. [原理说明](#原理说明)
3. [环境要求](#环境要求)
4. [数据准备](#数据准备)
5. [训练 SNN 模型](#训练-snn-模型)
6. [测试/推理](#测试推理)
7. [评估结果](#评估结果)
8. [在项目中使用](#在项目中使用)
9. [配置说明](#配置说明)
10. [常见问题](#常见问题)
11. [示例代码](#示例代码)

---

## 简介

SNN (Spiking Neural Network, 脉冲神经网络) 重建方案是一种基于脉冲神经网络的脉冲相机图像重建方法。该方法通过 SNN 学习从原始脉冲序列中提取时空特征，生成残差图像来增强传统的 TFP (Texture from Pulse) 重建结果。

### 核心特点

- **残差学习**：SNN 不直接生成完整图像，而是学习生成残差，叠加到 TFP 基础重建上
- **时序处理**：利用脉冲序列的时序信息，提取 8 个时间步的脉冲窗口
- **边缘增强**：通过 Sobel 边缘损失函数，重点优化图像边缘和细节
- **轻量级网络**：网络结构简单，训练和推理速度快

---

## 原理说明

### 重建流程

SNN 重建方案采用 **TFP + SNN 残差** 的混合架构：

```
最终重建图像 = TFP 基础重建 + SNN 预测的残差
```

1. **TFP 基础重建**：使用 `middleTFP` 算法从脉冲序列中计算基础亮度图像
2. **SNN 残差预测**：SNN 网络观察 8 个时间步的脉冲窗口，预测残差图像
3. **图像融合**：将 TFP 基础图像和 SNN 残差相加，得到最终重建结果

### 网络结构

```
输入: 脉冲窗口 [B, 8, 360, 640]
  ↓
Conv1 (1→32通道) + LIF1 (脉冲神经元)
  ↓
Conv2 (32→32通道) + LIF2 (脉冲神经元)
  ↓
Conv3 (32→1通道)
  ↓
输出: 残差图像 [B, 1, 360, 640]
```

### 损失函数

训练时使用组合损失函数：

```
Loss = L1_Loss(预测图像, GT图像) + 0.1 × Sobel_Edge_Loss(预测图像, GT图像)
```

- **L1 Loss**：保证整体像素值的准确性
- **Sobel Edge Loss**：通过 Sobel 算子提取边缘，保证边缘细节的还原

---

## 环境要求

### Python 依赖

```bash
torch >= 1.8.0
torchvision
numpy
opencv-python
snntorch          # 脉冲神经网络库
scikit-image      # 用于 PSNR/SSIM 计算
tqdm              # 进度条
```

### 安装 snntorch

```bash
pip install snntorch
```

---

## 数据准备

### 数据目录结构

SNN 训练需要以下数据目录结构：

```
GOPRO_Large_spike_seq/train/          # 脉冲数据根目录
├── GOPR0001/
│   └── spike/
│       ├── 000000.dat
│       ├── 000001.dat
│       └── ...
├── GOPR0002/
│   └── spike/
│       └── ...

GOPRO_Large/train/                    # GT 图像根目录
├── GOPR0001/
│   └── sharp/
│       ├── 000000.png
│       ├── 000001.png
│       └── ...
├── GOPR0002/
│   └── sharp/
│       └── ...
```

### 数据格式说明

- **脉冲文件 (.dat)**：二进制格式的脉冲流数据，每个文件包含多个时间步的脉冲帧
- **GT 图像 (.png)**：对应的清晰图像，灰度图，尺寸为 360×640
- **文件对应关系**：脉冲文件 `xxxxxx.dat` 对应 GT 图像 `xxxxxx.png`

---

## 训练 SNN 模型

### 方法一：命令行训练

```bash
python -m models.architectures.vrt.snn \
    --mode train \
    --root_spike GOPRO_Large_spike_seq/train \
    --root_gt GOPRO_Large/train \
    --epochs 100 \
    --batch_size 1 \
    --lr 2e-4 \
    --checkpoint_dir checkpoints \
    --device cuda:0 \
    --seed 42
```

### 方法二：Python 代码训练

```python
from models.architectures.vrt.snn import train

# 训练 SNN 模型
model = train(
    root_spike="GOPRO_Large_spike_seq/train",
    root_gt="GOPRO_Large/train",
    epochs=100,
    batch_size=1,
    lr=2e-4,
    checkpoint_dir="checkpoints",
    device="cuda:0",  # 或 "cpu"
    seed=42
)
```

### 训练参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_spike` | str | 必需 | 脉冲数据根目录 |
| `root_gt` | str | 必需 | GT 图像根目录 |
| `epochs` | int | 100 | 训练轮数 |
| `batch_size` | int | 1 | 批次大小 |
| `lr` | float | 2e-4 | 学习率 |
| `checkpoint_dir` | str | "checkpoints" | 检查点保存目录 |
| `device` | str | None | 设备（"cuda" 或 "cpu"），None 时自动检测 |
| `seed` | int | 42 | 随机种子 |

### 训练输出

训练过程中会：
- 每个 epoch 打印平均损失
- 每个 epoch 保存检查点：`checkpoints/snn_epoch_{epoch}.pth`
- 训练完成后，最终模型保存在：`checkpoints/snn_epoch_100.pth`

**训练日志示例：**
```
[Training] Using device: cuda:0
[GoProSpikeSNNDataset] Total samples: 1234
[Training] Checkpoints will be saved to: checkpoints
[Epoch 001/100] Loss: 0.123456
[Epoch 002/100] Loss: 0.098765
...
[Epoch 100/100] Loss: 0.045678
[Training] Training completed. Final checkpoint: checkpoints/snn_epoch_100.pth
```

---

## 测试/推理

### 方法一：命令行推理

```bash
python -m models.architectures.vrt.snn \
    --mode test \
    --root_spike GOPRO_Large_spike_seq/train \
    --checkpoint_path checkpoints/snn_epoch_100.pth \
    --output_dir output \
    --save_image \
    --device cuda:0
```

### 方法二：Python 代码推理

```python
from models.architectures.vrt.snn import test

# 测试/推理
test(
    root_spike_seq="GOPRO_Large_spike_seq/train",
    checkpoint_path="checkpoints/snn_epoch_100.pth",
    out_root="output",
    save_image=True,
    return_result=False,
    device="cuda:0"
)
```

### 推理参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_spike_seq` | str | 必需 | 脉冲数据根目录 |
| `checkpoint_path` | str | 必需 | 训练好的模型检查点路径 |
| `out_root` | str | None | 输出图像保存目录 |
| `save_image` | bool | False | 是否保存重建图像 |
| `return_result` | bool | False | 是否返回结果列表 |
| `device` | str | None | 设备（"cuda" 或 "cpu"） |

### 输出结果

如果 `save_image=True`，重建图像会保存到 `output/{seq}/{name}.png`

---

## 评估结果

### 方法一：命令行评估

```bash
python -m models.architectures.vrt.snn \
    --mode eval \
    --pred_root output \
    --gt_root GOPRO_Large/train \
    --device cuda:0
```

### 方法二：Python 代码评估

```python
from models.architectures.vrt.snn import evaluate

# 评估重建结果
metrics = evaluate(
    pred_root="output",
    gt_root="GOPRO_Large/train",
    device="cuda:0"
)
```

### 评估指标

评估函数会计算以下指标：

- **PSNR** (Peak Signal-to-Noise Ratio)：峰值信噪比，值越大越好
- **SSIM** (Structural Similarity Index)：结构相似性指数，值越大越好（0-1）
- **L1 Loss**：L1 损失，值越小越好
- **Edge Loss**：边缘损失，值越小越好
- **Total Loss**：总损失 = L1 Loss + 0.1 × Edge Loss

### 评估输出示例

```
[Evaluation] Using device: cuda:0
Evaluating sequences: 100%|████████| 10/10 [00:30<00:00,  3.05s/it]

===== Sequence Evaluation =====
GOPR0001 | PSNR: 28.45 | SSIM: 0.8234 | Total Loss: 0.045678
GOPR0002 | PSNR: 29.12 | SSIM: 0.8456 | Total Loss: 0.042345
...

===== Best Sequences =====
Best PSNR sequence: GOPR0005 -> PSNR: 30.23
Best SSIM sequence: GOPR0003 -> SSIM: 0.8567
```

---

## 在项目中使用

### 在数据集配置中使用

在项目的配置文件中（如 `options/gopro_rgbspike_local.json`），可以配置使用 SNN 重建方案：

```json
{
  "datasets": {
    "train": {
      "spike_reconstruction": {
        "type": "snn",
        "middle_tfp_center": 44,
        "checkpoint_path": "checkpoints/snn_epoch_100.pth",
        "spike_win": 8,
        "device": "cuda:0"
      }
    }
  }
}
```

### 直接使用 SNNReconstructor

```python
from data.spike_recc.snn.reconstructor import SNNReconstructor
import numpy as np

# 初始化重建器
reconstructor = SNNReconstructor(
    checkpoint_path="checkpoints/snn_epoch_100.pth",
    spike_win=8,
    center=44,
    device="cuda:0"
)

# 重建单个脉冲序列
spike_data = load_vidar_dat("path/to/spike.dat", width=640, height=360)  # [T, H, W]
reconstructed_image = reconstructor(spike_data)  # [H, W], 归一化到 [0, 1]
```

---

## 配置说明

### 配置文件参数

在项目配置文件中，SNN 重建方案支持以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | str | "snn" | 重建方案类型，必须为 "snn" |
| `checkpoint_path` | str | 必需 | SNN 模型检查点路径 |
| `spike_win` | int | 8 | 脉冲窗口大小（时间步数） |
| `middle_tfp_center` | int | 44 | TFP 重建的中心时间步 |
| `device` | str | "cpu" | 推理设备（"cuda:0" 或 "cpu"） |

### 配置示例

**简单配置（字符串形式）：**
```json
"spike_reconstruction": "snn"
```

**详细配置（字典形式）：**
```json
"spike_reconstruction": {
  "type": "snn",
  "checkpoint_path": "checkpoints/snn_epoch_100.pth",
  "spike_win": 8,
  "middle_tfp_center": 44,
  "device": "cuda:0"
}
```

---

## 常见问题

### Q1: 训练时出现 "CUDA out of memory" 错误

**解决方案：**
- 减小 `batch_size`（默认已经是 1，可以尝试更小的值）
- 使用 CPU 训练：`--device cpu`
- 减少训练数据量

### Q2: 检查点文件找不到

**解决方案：**
- 检查 `checkpoint_path` 路径是否正确
- 确保训练已完成并生成了检查点文件
- 使用绝对路径而不是相对路径

### Q3: 重建结果质量不好

**可能原因和解决方案：**
- **训练不充分**：增加训练轮数 `--epochs`
- **学习率不合适**：尝试调整学习率 `--lr`
- **数据质量问题**：检查脉冲数据和 GT 图像是否匹配
- **模型未收敛**：查看训练损失曲线，确保损失持续下降

### Q4: 如何选择最佳检查点？

**建议：**
- 训练多个 epoch，保存所有检查点
- 使用评估模式测试不同 epoch 的模型
- 选择 PSNR 和 SSIM 最高的检查点

### Q5: 脉冲窗口大小可以调整吗？

**说明：**
- 默认 `spike_win=8` 是经过实验验证的最佳值
- 可以尝试其他值（如 4, 6, 10），但需要重新训练模型
- 修改后需要同时更新训练代码和推理代码中的 `SPIKE_WIN` 常量

### Q6: SNN 和 TFP 的区别是什么？

**区别：**
- **TFP**：纯传统算法，直接从脉冲序列计算图像，速度快但质量一般
- **SNN**：深度学习增强方案，在 TFP 基础上用 SNN 学习残差，质量更好但需要训练

**选择建议：**
- 如果追求速度：使用 `middle_tfp`
- 如果追求质量：使用 `snn`
- 如果追求平衡：使用 `spikecv_tfp`（多通道 TFP）

---

## 示例代码

### 完整训练-测试-评估流程

```python
from models.architectures.vrt.snn import train, test, evaluate

# 1. 训练
print("开始训练 SNN 模型...")
train(
    root_spike="GOPRO_Large_spike_seq/train",
    root_gt="GOPRO_Large/train",
    epochs=100,
    batch_size=1,
    lr=2e-4,
    checkpoint_dir="checkpoints",
    device="cuda:0"
)

# 2. 测试/推理
print("开始测试...")
test(
    root_spike_seq="GOPRO_Large_spike_seq/train",
    checkpoint_path="checkpoints/snn_epoch_100.pth",
    out_root="output",
    save_image=True,
    device="cuda:0"
)

# 3. 评估
print("开始评估...")
metrics = evaluate(
    pred_root="output",
    gt_root="GOPRO_Large/train",
    device="cuda:0"
)

# 打印每个序列的指标
for seq, m in metrics.items():
    print(f"{seq}: PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.4f}")
```

### 在数据加载中使用

```python
from data.spike_recc.snn.reconstructor import SNNReconstructor
from data.spike_recc import SpikeStream

# 初始化重建器
snn_reconstructor = SNNReconstructor(
    checkpoint_path="checkpoints/snn_epoch_100.pth",
    spike_win=8,
    center=44,
    device="cuda:0"
)

# 加载脉冲数据
spike_stream = SpikeStream(
    offline=True,
    filepath="path/to/spike.dat",
    spike_h=360,
    spike_w=640
)
spike_matrix = spike_stream.get_spike_matrix(flipud=True)  # [T, H, W]

# 重建图像
reconstructed = snn_reconstructor(spike_matrix)  # [H, W], [0, 1]
```

### 批量处理

```python
import os
from data.spike_recc.snn.reconstructor import SNNReconstructor
from data.spike_recc.middle_tfp.spike_utils import load_vidar_dat
import cv2

reconstructor = SNNReconstructor(
    checkpoint_path="checkpoints/snn_epoch_100.pth",
    device="cuda:0"
)

input_dir = "input_spikes"
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".dat"):
        # 加载脉冲
        spike_path = os.path.join(input_dir, filename)
        spike = load_vidar_dat(spike_path, width=640, height=360)
        
        # 重建
        image = reconstructor(spike)
        
        # 保存
        output_path = os.path.join(output_dir, filename.replace(".dat", ".png"))
        cv2.imwrite(output_path, (image * 255).astype(np.uint8))
```

---

## 技术细节

### 网络架构

- **输入维度**：`[Batch, 8, 360, 640]` - 8 个时间步的脉冲窗口
- **输出维度**：`[Batch, 1, 360, 640]` - 单通道残差图像
- **参数量**：约 10K 参数（轻量级网络）

### 训练策略

- **优化器**：Adam
- **学习率**：2e-4（固定）
- **损失权重**：L1 Loss (1.0) + Edge Loss (0.1)
- **批次大小**：1（由于脉冲序列较长）

### 性能优化

- 使用 `torch.no_grad()` 进行推理加速
- 支持 GPU 加速训练和推理
- 批量处理时建议使用 DataLoader

---

## 更新日志

- **v1.0** (2024): 初始版本，支持训练、测试和评估功能
- 集成到项目统一配置系统
- 支持命令行和 Python API 两种使用方式

---

## 参考

- SNN 训练代码：`models/architectures/vrt/snn.py`
- SNN 重建器：`data/spike_recc/snn/reconstructor.py`
- TFP 工具函数：`data/spike_recc/middle_tfp/spike_utils.py`

---

## 联系方式

如有问题或建议，请参考项目主文档或提交 Issue。
