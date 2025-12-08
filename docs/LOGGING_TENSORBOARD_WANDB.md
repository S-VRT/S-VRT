# TensorBoard、Weights & Biases (WANDB) 与 SwanLab 日志记录指南

本文档介绍如何在KAIR项目中使用TensorBoard、Weights & Biases (WANDB)以及SwanLab进行训练日志记录和实验跟踪。

## 目录

- [概述](#概述)
- [安装依赖](#安装依赖)
- [配置说明](#配置说明)
- [TensorBoard使用](#tensorboard使用)
- [WANDB使用](#wandb使用)
- [SwanLab使用](#swanlab使用)
- [非交互式训练](#非交互式训练)
- [日志内容](#日志内容)
- [常见问题](#常见问题)

## 概述

KAIR项目集成了三种主要的日志记录工具：

1. **TensorBoard**: 用于本地可视化和监控训练过程
2. **Weights & Biases (WANDB)**: 用于云端实验跟踪、协作和模型管理
3. **SwanLab**: 支持云端与本地一体化的实验跟踪平台，提供国内可访问的可视化体验

三种工具可以同时使用，也可以按需单独启用。所有日志记录功能都是可选的，可以通过配置文件轻松启用或禁用。

## 安装依赖

### TensorBoard

```bash
pip install tensorboard
```

### Weights & Biases

```bash
pip install wandb
```

### SwanLab

```bash
pip install swanlab
```

或者使用项目提供的requirements.txt：

```bash
pip install -r requirement.txt
```

## 配置说明

在训练配置JSON文件中，添加`logging`部分来配置日志记录：

```json
{
  "logging": {
    "use_tensorboard": true,
    "use_wandb": true,
    "wandb_api_key": "your-wandb-api-key-here",
    "wandb_project": "VRT-VideoDeblurring",
    "wandb_entity": null,
    "wandb_name": null,
    "use_swanlab": false,
    "swanlab_api_key": null,
    "swanlab_project": "VRT-VideoDeblurring",
    "swanlab_workspace": null,
    "swanlab_name": null,
    "swanlab_description": null,
    "swanlab_mode": null
  }
}
```

### 配置参数说明

#### TensorBoard配置

- `use_tensorboard` (bool): 是否启用TensorBoard日志记录
  - `true`: 启用TensorBoard
  - `false`: 禁用TensorBoard（默认）

#### WANDB配置

- `use_wandb` (bool): 是否启用WANDB日志记录
  - `true`: 启用WANDB
  - `false`: 禁用WANDB（默认）

- `wandb_api_key` (string | null): WANDB API密钥
  - 设置为你的API密钥字符串以实现非交互式登录
  - 设置为`null`时，会尝试从环境变量`WANDB_API_KEY`读取
  - 如果两者都不存在，WANDB将以离线模式运行

- `wandb_project` (string | null): WANDB项目名称
  - 指定要将实验记录到的项目名称
  - 如果项目不存在，WANDB会自动创建
  - 设置为`null`时使用默认项目名称`kair-training`

- `wandb_entity` (string | null): WANDB团队/用户名
  - 指定项目所属的团队或用户
  - 设置为`null`时使用你的个人账户

- `wandb_name` (string | null): 实验运行名称
  - 为当前训练运行指定一个名称
  - 设置为`null`时自动使用任务名称（`task`字段的值）

#### SwanLab配置

- `use_swanlab` (bool): 是否启用SwanLab日志记录
  - `true`: 启用SwanLab
  - `false`: 禁用SwanLab（默认）

- `swanlab_api_key` (string | null): SwanLab API密钥
  - 设置为你的API密钥字符串以实现非交互式登录
  - 设置为`null`时，会尝试从环境变量`SWANLAB_API_KEY`读取，或依赖`swanlab login`
  - 如果两者都不存在，SwanLab会自动切换到离线/本地模式

- `swanlab_project` (string | null): SwanLab项目名称
  - 指定要记录的项目，未填写时默认使用任务名
  - 如项目不存在，SwanLab会自动创建

- `swanlab_workspace` (string | null): SwanLab工作空间（用户名或团队名）
  - 为空时使用当前登录用户

- `swanlab_name` (string | null): SwanLab实验名称
  - 不填写时自动使用任务名

- `swanlab_description` (string | null): 可选的实验说明文本

- `swanlab_mode` (string | null): SwanLab运行模式
  - 可设置为`"cloud"`、`"offline"`、`"local"`或`"disabled"`
  - 留空时会根据API密钥自动推断（有密钥则云端，否则离线）
- `swanlab_auto_resume` (bool): 是否自动缓存/复用 cloud run id（默认 `true`）
  - 设置为 `false` 时，每次启动都会创建新 run
- `swanlab_run_id_file` (string | null): 缓存 run id 的文件路径
  - 默认是 `experiments/{task_name}/swanlab_run.id`
- `swanlab_run_id` (string | null): 手动指定 run id（适合多机协同）
- `swanlab_resume_strategy` (string | null): 传递给 `swanlab.init` 的 `resume` 策略
  - 默认为 `"allow"`（同名 run 存在则续写，不存在则新建）

### 配置示例

#### 示例1: 只启用TensorBoard

```json
{
  "logging": {
    "use_tensorboard": true,
    "use_wandb": false
  }
}
```

#### 示例2: 只启用WANDB（使用环境变量中的API密钥）

```json
{
  "logging": {
    "use_tensorboard": false,
    "use_wandb": true,
    "wandb_api_key": null,
    "wandb_project": "MyProject",
    "wandb_entity": "my-team",
    "wandb_name": "experiment-v1"
  }
}
```

#### 示例3: 同时启用TensorBoard和WANDB（非交互式）

```json
{
  "logging": {
    "use_tensorboard": true,
    "use_wandb": true,
    "wandb_api_key": "your-api-key-here",
    "wandb_project": "VRT-Training",
    "wandb_entity": null,
    "wandb_name": null
  }
}
```

#### 示例4: 启用SwanLab（离线模式）

```json
{
  "logging": {
    "use_tensorboard": false,
    "use_wandb": false,
    "use_swanlab": true,
    "swanlab_api_key": null,
    "swanlab_project": "VRT-Training",
    "swanlab_name": "experiment-v1",
    "swanlab_mode": "offline"
  }
}
```

## TensorBoard使用

### 启动TensorBoard

训练开始后，TensorBoard日志会自动保存到以下目录：

```
experiments/{task_name}/tensorboard/
```

要查看TensorBoard，在终端运行：

```bash
tensorboard --logdir experiments/{task_name}/tensorboard
```

然后在浏览器中打开 `http://localhost:6006`

### TensorBoard记录的内容

- **训练指标**:
  - 训练损失（`train/loss`）
  - 学习率（`train/learning_rate`）
  - 其他训练指标

- **测试指标**:
  - PSNR（`test/psnr`）
  - SSIM（`test/ssim`）
  - PSNR_Y（`test/psnr_y`）
  - SSIM_Y（`test/ssim_y`）

- **图像**:
  - 测试图像（如果启用）

### TensorBoard界面功能

- **SCALARS**: 查看训练和测试指标的时间序列图
- **IMAGES**: 查看测试图像的可视化
- **GRAPHS**: 查看模型计算图（如果记录）

## WANDB使用

### 获取WANDB API密钥

1. 访问 [https://wandb.ai](https://wandb.ai) 并注册/登录账户
2. 进入 [Settings → API keys](https://wandb.ai/settings)
3. 复制你的API密钥

### 交互式登录（可选）

如果你想使用交互式登录而不是在配置文件中存储API密钥：

```bash
wandb login
```

然后输入你的API密钥。登录后，即使配置文件中`wandb_api_key`为`null`，WANDB也能正常工作。

### 查看WANDB仪表板

训练开始后，访问 [https://wandb.ai](https://wandb.ai) 并登录你的账户，即可看到：

- 实时训练指标图表
- 测试结果和图像
- 超参数配置
- 系统资源使用情况（GPU、CPU、内存）

### WANDB记录的内容

- **训练指标**:
  - 训练损失（`train/loss`）
  - 学习率（`train/learning_rate`）
  - 其他训练指标

- **测试指标**:
  - PSNR（`test/psnr`）
  - SSIM（`test/ssim`）
  - PSNR_Y（`test/psnr_y`）
  - SSIM_Y（`test/ssim_y`）

- **图像**:
  - 测试图像（如果启用）

- **配置信息**:
  - 完整的训练配置（模型参数、优化器设置等）

### WANDB离线模式

如果没有提供API密钥且环境变量中也没有，WANDB会自动切换到离线模式。离线模式下：

- 所有数据保存在本地：`wandb/offline-run-{timestamp}/`
- 训练结束后，可以使用以下命令同步到云端：

```bash
wandb sync wandb/offline-run-{timestamp}/
```

## SwanLab使用

### 获取SwanLab API密钥

1. 访问 [https://swanlab.cn](https://swanlab.cn) 并注册/登录账户
2. 打开「个人设置 → API Key」
3. 复制你的API密钥

### 登录与模式切换

- **命令行登录**（推荐）：

  ```bash
  swanlab login
  ```

- **环境变量**：`export SWANLAB_API_KEY="your-api-key"`
- **命令切换模式**：`swanlab online` / `swanlab offline` / `swanlab local`

配置文件中的`swanlab_api_key`也会在训练脚本启动时自动注入`SWANLAB_API_KEY`环境变量，实现非交互式登录。

### 查看SwanLab仪表板

训练开始后，可访问 [https://swanlab.cn](https://swanlab.cn) 并进入对应 Workspace/Project，或在本地运行：

```bash
swanlab watch
```

SwanLab 支持云端仪表板与本地 Dashboard（需安装 `swanlab-dashboard`），适合在离线环境中浏览日志。

### SwanLab记录的内容

- **训练/测试指标**：与 WANDB/TensorBoard 保持一致（如 `train/loss`、`test/psnr` 等）
- **图像**：自动上传 `log_images` 中记录的样本
- **配置**：`swanlab_run.config.update` 会同步完整配置，便于对比
- **系统信息**：自动记录 GPU/CPU/内存占用等

### SwanLab断点续炼/自动续写

- 当 `use_swanlab` 为 `true` 且运行在云端模式时，日志器会自动将当前 run 的 `id` 缓存到 `experiments/{task_name}/swanlab_run.id`
- 训练中断（例如 30k 计划在 10k 处停机）后，只要继续使用同一任务目录/配置文件启动训练，就会自动读取该文件并传入 `swanlab.init(id=..., resume="allow")`，因此后续日志会被写入同一个 run
- 如果需要强制开启一个全新的 run，可任选其一：
  1. 删除 `experiments/{task_name}/swanlab_run.id`
  2. 在配置中设置 `"swanlab_auto_resume": false`
  3. 手动指定新的 `swanlab_run_id`
- `swanlab_run_id_file` 可改成自定义路径；`swanlab_resume_strategy` 也可以切换为 `"must"`（必须存在 run 才能续写）或 `"never"`（总是新 run）
- 在 `offline/local` 模式下不会传入 `resume` 参数，run id 文件仅用于记录，不会触发云端续写

### SwanLab离线与本地模式

- 将`swanlab_mode`设为`"offline"`或`"local"`即可在无网络环境下运行
- 离线数据默认保存在`./swanlog/offline-run-{timestamp}/`
- 可随时使用 `swanlab offline` / `swanlab local` CLI 强制切换模式
- 训练结束后通过 `swanlab sync swanlog/offline-run-{timestamp}/` 同步到云端

## 非交互式训练

对于服务器训练或自动化脚本，推荐使用非交互式模式。WANDB 与 SwanLab 都支持以下方式：

### WANDB - 方式1: 在配置文件中设置API密钥（推荐）

在配置JSON文件中直接设置`wandb_api_key`：

```json
{
  "logging": {
    "use_wandb": true,
    "wandb_api_key": "your-api-key-here",
    "wandb_project": "MyProject"
  }
}
```

**注意**: 请确保配置文件不会被提交到公共代码仓库。建议使用`.gitignore`排除包含敏感信息的配置文件。

### WANDB - 方式2: 使用环境变量

在启动训练前设置环境变量：

```bash
export WANDB_API_KEY="your-api-key-here"
python main_train_vrt.py --opt options/vrt/your_config.json
```

或者在单行命令中：

```bash
WANDB_API_KEY="your-api-key-here" python main_train_vrt.py --opt options/vrt/your_config.json
```

### SwanLab - 方式1: 在配置文件中设置API密钥

```json
{
  "logging": {
    "use_swanlab": true,
    "swanlab_api_key": "your-api-key-here",
    "swanlab_project": "MyProject",
    "swanlab_mode": "cloud"
  }
}
```

### SwanLab - 方式2: 使用环境变量或CLI

```bash
export SWANLAB_API_KEY="your-api-key-here"
swanlab login  # 可选，确认登录状态
python main_train_vrt.py --opt options/vrt/your_config.json
```

也可以直接使用命令切换模式：

```bash
swanlab offline   # 强制离线记录
swanlab online    # 切换回云端上传
```

### 安全建议

1. **不要将API密钥提交到Git**: 在配置文件中使用API密钥时，确保该配置文件在`.gitignore`中
2. **使用环境变量**: 在CI/CD或共享服务器上，优先使用环境变量
3. **使用密钥管理工具**: 在生产环境中，考虑使用密钥管理服务（如AWS Secrets Manager、HashiCorp Vault等）

## 日志内容

### 训练日志频率

日志记录的频率由配置文件中的以下参数控制：

- `checkpoint_print`: 训练指标记录的间隔（默认：200次迭代）
- `checkpoint_test`: 测试指标记录的间隔（默认：5000次迭代）

### 记录的指标

#### 训练阶段（每个`checkpoint_print`间隔）

- `train/loss`: 训练损失
- `train/learning_rate`: 当前学习率
- 其他模型特定的训练指标

#### 测试阶段（每个`checkpoint_test`间隔）

- `test/psnr`: 平均PSNR值
- `test/ssim`: 平均SSIM值
- `test/psnr_y`: Y通道平均PSNR值
- `test/ssim_y`: Y通道平均SSIM值
- 测试图像（如果启用）

## 常见问题

### Q1: TensorBoard显示"No dashboards are active"

**原因**: TensorBoard日志目录不存在或为空。

**解决方案**:
1. 确保训练已经开始并至少完成了一次`checkpoint_print`迭代
2. 检查日志目录路径是否正确
3. 确认`use_tensorboard`在配置中设置为`true`

### Q2: WANDB初始化失败

**可能原因**:
1. 未安装wandb包
2. API密钥无效或未设置
3. 网络连接问题

**解决方案**:
1. 安装wandb: `pip install wandb`
2. 检查API密钥是否正确设置
3. 检查网络连接，或使用离线模式

### Q3: SwanLab初始化失败

**可能原因**:
1. 未安装`swanlab`包
2. 未登录或缺少 `swanlab_api_key` / `SWANLAB_API_KEY`
3. 选择了`cloud`模式但当前网络无法访问 SwanLab

**解决方案**:
1. 安装依赖：`pip install swanlab`
2. 使用 `swanlab login` 或在配置/环境变量中提供 API key
3. 在无法联网的环境中使用 `swanlab_mode: "offline"` 或运行 `swanlab offline`

### Q4: 如何在分布式训练中使用日志记录？

日志记录只在rank 0进程中启用，避免重复记录。这是自动处理的，无需额外配置。

### Q5: WANDB离线模式的数据在哪里？

离线数据保存在训练目录下的`wandb/offline-run-{timestamp}/`文件夹中。可以使用`wandb sync`命令同步到云端。

### Q6: 如何禁用所有日志记录？

在配置文件中设置：

```json
{
  "logging": {
    "use_tensorboard": false,
    "use_wandb": false,
    "use_swanlab": false
  }
}
```

或者完全移除`logging`部分（将使用默认值，即全部禁用）。

### Q7: 日志文件占用太多磁盘空间怎么办？

- **TensorBoard**: 定期清理旧的日志目录
- **WANDB**: 在WANDB网页界面中删除不需要的运行，或使用WANDB的保留策略

### Q8: 如何查看历史训练记录的TensorBoard日志？

```bash
tensorboard --logdir experiments/{task_name}/tensorboard
```

即使训练已经结束，只要日志文件存在，就可以查看。

## 总结

KAIR项目的日志记录系统提供了灵活且强大的实验跟踪功能：

- ✅ 支持TensorBoard、WANDB与SwanLab，可单独或组合使用
- ✅ 非交互式训练支持（通过配置文件或环境变量）
- ✅ 自动记录训练和测试指标
- ✅ 分布式训练自动处理
- ✅ 易于配置和禁用

通过合理使用这些工具，你可以更好地监控训练过程、比较不同实验、并管理模型版本。

