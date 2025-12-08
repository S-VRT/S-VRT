# 使用 launch_test.sh 进行评估

## 概述

`launch_test.sh` 是用于运行 VRT/S-VRT 模型测试的启动脚本。它支持使用训练时的 JSON 配置文件进行评估。

## 基本用法

### 1. 配置模型路径

在训练用的 JSON 配置文件中（如 `options/gopro_rgbspike_local.json`），设置模型路径：

```json
"path": {
  "root": "experiments",
  "pretrained_netG": "experiments/gopro_rgbspike_channel_concat/models/30000_G.pth",
  "pretrained_netE": "experiments/gopro_rgbspike_channel_concat/models/30000_E.pth"
}
```

**注意：**
- `pretrained_netE` 是 EMA（指数移动平均）模型，测试时会**优先使用 E 模型**
- 如果 E 模型不存在，会自动使用 G 模型
- 模型路径可以使用相对路径或绝对路径

### 2. 运行测试

```bash
./launch_test.sh 1 options/gopro_rgbspike_local.json
```

**参数说明：**
- `1`: GPU 数量（单 GPU 测试）
- `options/gopro_rgbspike_local.json`: 训练时使用的配置文件

### 3. 多 GPU 测试

```bash
./launch_test.sh 3 options/gopro_rgbspike_local.json
```

使用 3 个 GPU 进行测试。

### 4. 指定特定 GPU

```bash
./launch_test.sh --gpus 0,1,2 options/gopro_rgbspike_local.json
```

## 完整示例

评估 30000 迭代的模型：

```bash
# 1. 确保配置文件中的模型路径已设置
# options/gopro_rgbspike_local.json 中：
#   "pretrained_netG": "experiments/gopro_rgbspike_channel_concat/models/30000_G.pth"
#   "pretrained_netE": "experiments/gopro_rgbspike_channel_concat/models/30000_E.pth"

# 2. 运行测试（单 GPU）
./launch_test.sh 1 options/gopro_rgbspike_local.json

# 或使用多 GPU
./launch_test.sh 3 options/gopro_rgbspike_local.json
```

## 模型加载优先级

1. **优先使用 E 模型（EMA）**：如果 `pretrained_netE` 存在且文件存在，会优先加载 E 模型
2. **回退到 G 模型**：如果 E 模型不存在，会使用 G 模型
3. **错误提示**：如果两个模型都不存在，会显示错误信息

## 输出结果

测试结果会保存在：
- `results/{task_name}/` - 测试图像结果
- 控制台输出 - PSNR、SSIM 等指标

## 其他选项

查看所有选项：

```bash
./launch_test.sh --help
```

常用选项：
- `--prepare-data`: 运行数据准备脚本
- `--dataset-root PATH`: 指定数据集根目录
- `--gopro-root PATH`: 覆盖 GoPro 数据集路径
- `--spike-root PATH`: 覆盖 Spike 数据集路径
