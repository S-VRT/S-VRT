# VRT 分布式训练指南

## 概述

本项目已完成现代化分布式训练改造，支持以下两种场景：

1. **平台 DDP 训练**：云平台/集群自动注入环境变量
2. **本地多卡训练**：使用 `torchrun` 或单卡训练

核心特性：
- ✅ 自动检测分布式模式（基于 `WORLD_SIZE` 环境变量）
- ✅ 支持 PyTorch 原生 `env://` 初始化方式
- ✅ 兼容 SLURM 集群环境
- ✅ 正确的设备分配（基于 `LOCAL_RANK`）
- ✅ 自动数据分片（DistributedSampler）
- ✅ 仅在主进程保存模型和日志

---

## 快速开始

### 场景一：平台 DDP 训练

**适用情况**：云平台（如阿里云、腾讯云等）已为每个进程自动注入环境变量。

平台会设置以下环境变量：
- `RANK`: 全局进程序号
- `LOCAL_RANK`: 节点内进程序号
- `WORLD_SIZE`: 总进程数
- `MASTER_ADDR`: 主节点地址
- `MASTER_PORT`: 主节点端口

**运行命令**：

```bash
python -u main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

⚠️ **重要**：不要使用 `torchrun`！平台已经为每个进程运行了相同的命令，使用 `torchrun` 会导致嵌套多进程。

---

### 场景二：本地多卡训练

**适用情况**：在本地机器或未配置环境变量的服务器上训练。

#### 方式 1：使用启动脚本（推荐）

```bash
# 单卡训练
./launch_train.sh 1

# 4 卡训练
./launch_train.sh 4

# 8 卡训练，使用自定义配置
./launch_train.sh 8 options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

#### 方式 2：直接使用 torchrun

```bash
# 4 卡训练
torchrun --nproc_per_node=4 main_train_vrt.py \
    --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json

# 8 卡训练
torchrun --nproc_per_node=8 main_train_vrt.py \
    --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

#### 方式 3：单卡训练

```bash
python main_train_vrt.py --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

---

## 配置文件说明

### 分布式相关配置项

在 JSON 配置文件中（如 `006_train_vrt_videodeblurring_gopro_rgbspike.json`）：

```json
{
  "gpu_ids": [0,1,2],
  "dist": true,
  "find_unused_parameters": false,
  "use_static_graph": true
}
```

#### `gpu_ids` 字段

- **单进程模式**：指定使用哪些 GPU（例如 `[0,1,2]` 表示使用 0、1、2 号卡）
- **分布式模式**：此字段被忽略，设备由 `torchrun` 或平台自动分配
- **单卡训练**：设置为 `[0]`

程序会自动设置 `CUDA_VISIBLE_DEVICES` 环境变量（仅在单进程模式下）。

#### `dist` 字段

- 自动从 `WORLD_SIZE` 环境变量检测
- 手动设置会被自动检测结果覆盖
- 保留此字段是为了文档目的

#### `find_unused_parameters` 字段

- DDP 相关参数，控制是否查找未使用的模型参数
- VRT 模型设置为 `false` 可提升性能

#### `use_static_graph` 字段

- 启用静态计算图优化（PyTorch >= 1.11）
- 对固定网络结构的模型可提升训练速度

---

## 环境变量

### 自动设置的环境变量

在分布式训练中，以下环境变量由 `torchrun` 或平台自动设置：

| 环境变量 | 说明 | 示例值 |
|---------|------|--------|
| `RANK` | 全局进程序号（0 到 world_size-1） | `0`, `1`, `2`, ... |
| `LOCAL_RANK` | 单节点内进程序号 | `0`, `1`, `2`, ... |
| `WORLD_SIZE` | 总进程数（等于 GPU 数量） | `4`, `8` |
| `MASTER_ADDR` | 主节点地址 | `localhost`, `192.168.1.100` |
| `MASTER_PORT` | 主节点端口 | `29500` |

### 推荐的 NCCL 环境变量

为了提升训练稳定性和性能，建议设置以下环境变量：

```bash
# 异步错误处理（推荐）
export NCCL_ASYNC_ERROR_HANDLING=1

# 如果没有 InfiniBand 网络，禁用 IB
export NCCL_IB_DISABLE=1

# 限制 CUDA 连接数（某些模型可提升稳定性）
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 启用 NCCL 调试信息（调试时使用）
# export NCCL_DEBUG=INFO
```

可以将这些变量添加到 `~/.bashrc` 或在运行训练前导出：

```bash
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
./launch_train.sh 4
```

---

## 工作原理

### 自动检测机制

程序启动时按以下顺序检测分布式模式：

1. 检查 `WORLD_SIZE` 环境变量
   - 如果 `WORLD_SIZE > 1`：启用分布式训练
   - 如果 `WORLD_SIZE` 不存在或等于 1：单进程模式

2. 读取 rank 信息（优先级从高到低）：
   - `LOCAL_RANK` 环境变量（torchrun 标准）
   - `SLURM_LOCALID`（SLURM 集群）
   - 默认值：0

3. 设置 CUDA 设备：
   ```python
   torch.cuda.set_device(local_rank)
   device = torch.device(f"cuda:{local_rank}")
   ```

4. 初始化进程组：
   ```python
   torch.distributed.init_process_group(
       backend='nccl',
       init_method='env://'
   )
   ```

### 数据加载

在分布式模式下，使用 `DistributedSampler` 自动分片数据：

```python
from torch.utils.data.distributed import DistributedSampler

# 创建 sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers
)

# 每个 epoch 开始时设置 epoch（确保数据随机性）
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        # 训练代码
        ...
```

### 模型封装

模型自动使用 `DistributedDataParallel` 封装：

```python
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=False,  # 性能优化
    find_unused_parameters=opt['find_unused_parameters']
)
```

### 日志和保存

**重要原则**：只在主进程（rank 0）执行以下操作：

- ✅ 创建日志目录
- ✅ 保存模型检查点
- ✅ 保存训练状态
- ✅ 写入 TensorBoard/W&B 日志
- ✅ 打印训练信息

代码示例：

```python
from utils.utils_dist import is_main_process

if is_main_process():
    logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
    torch.save(model.state_dict(), 'checkpoint.pth')
```

---

## 常见问题

### 1. 为什么不能在平台 DDP 中使用 torchrun？

**原因**：平台已经为每个 GPU 启动了一个独立的进程，并为每个进程设置了正确的环境变量。如果再使用 `torchrun`，会导致每个进程又创建多个子进程，造成混乱。

**正确做法**：
```bash
# 平台 DDP（每个进程运行相同命令）
python -u main_train_vrt.py --opt config.json
```

**错误做法**：
```bash
# ❌ 会导致嵌套多进程
torchrun --nproc_per_node=4 main_train_vrt.py --opt config.json
```

### 2. 如何确认分布式训练正常工作？

查看训练日志，应该看到：

```
========================================
Distributed Training Setup
========================================
Backend: nccl
World Size: 4
Rank: 0
Local Rank: 0
Master: localhost:29500
========================================
```

每个进程会输出自己的 rank 信息。只有 rank 0 会保存模型和输出详细日志。

### 3. CUDA out of memory 错误

**原因**：批次大小（batch size）需要根据 GPU 数量调整。

**解决方案**：

- 分布式训练时，每张卡的批次大小 = 配置中的 `dataloader_batch_size`
- 总批次大小 = `dataloader_batch_size × 卡数`

例如：
- 单卡训练：`batch_size = 8`
- 4 卡训练：每卡 `batch_size = 2`（总批次 = 2 × 4 = 8）

在配置文件中调整：

```json
{
  "datasets": {
    "train": {
      "dataloader_batch_size": 2  // 减小每卡批次大小
    }
  }
}
```

### 4. NCCL 初始化超时

**常见原因**：

1. 防火墙阻止进程间通信
2. `MASTER_ADDR` 或 `MASTER_PORT` 设置错误
3. 网络配置问题

**解决方案**：

```bash
# 增加超时时间
export NCCL_TIMEOUT=1800

# 启用调试信息
export NCCL_DEBUG=INFO

# 指定网络接口（如果有多个网卡）
export NCCL_SOCKET_IFNAME=eth0

# 重新运行训练
./launch_train.sh 4
```

### 5. 不同进程的损失值不同步

这是正常现象！每个进程处理不同的数据分片，因此：

- ✅ 每个 GPU 的损失值可能不同（数据不同）
- ✅ 梯度会在反向传播时自动同步
- ✅ 模型参数在所有 GPU 上保持一致

如需同步指标用于日志记录，使用：

```python
from utils.utils_dist import all_reduce_mean

# 计算所有进程的平均损失
avg_loss = all_reduce_mean(loss_tensor)
```

### 6. 训练速度没有线性提升

**正常情况**：

- 2 卡理论上应该快 2 倍，但实际约 1.7-1.9 倍
- 4 卡理论上应该快 4 倍，但实际约 3.2-3.6 倍
- 8 卡理论上应该快 8 倍，但实际约 6-7 倍

**原因**：

1. 通信开销（梯度同步）
2. I/O 瓶颈（数据加载）
3. 负载不均衡

**优化建议**：

```json
{
  "datasets": {
    "train": {
      "dataloader_num_workers": 8,  // 增加数据加载线程
      "dataloader_batch_size": 4    // 增大批次减少通信频率
    }
  }
}
```

---

## SLURM 集群支持

本项目兼容 SLURM 作业调度系统。

### SLURM 任务脚本示例

创建 `submit_job.sh`：

```bash
#!/bin/bash
#SBATCH --job-name=vrt_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 加载环境
module load cuda/11.8
module load pytorch/2.0

# NCCL 配置
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

# 运行训练（SLURM 会自动设置环境变量）
srun python -u main_train_vrt.py \
    --opt options/vrt/006_train_vrt_videodeblurring_gopro_rgbspike.json
```

提交任务：

```bash
sbatch submit_job.sh
```

SLURM 会自动设置：
- `SLURM_PROCID` → 映射到 `RANK`
- `SLURM_LOCALID` → 映射到 `LOCAL_RANK`
- `SLURM_NTASKS` → 映射到 `WORLD_SIZE`

---

## 监控训练

### TensorBoard

训练日志自动保存到 `experiments/<task_name>/tb_logger/`：

```bash
tensorboard --logdir experiments/006_train_vrt_videodeblurring_gopro_rgbspike/tb_logger
```

### Weights & Biases

如果配置了 W&B：

```json
{
  "logging": {
    "use_wandb": true,
    "wandb_project": "VRT-VideoDeblurring",
    "wandb_api_key": "your_api_key"
  }
}
```

训练会自动上传到 W&B 平台。

### 命令行输出

只有主进程（rank 0）会输出详细信息：

```
[2025-11-06 10:00:00] Epoch: 1, Iter: 100, Loss: 0.0234
[2025-11-06 10:05:00] Epoch: 1, Iter: 200, Loss: 0.0198
```

其他进程会保持静默或只输出关键信息。

---

## 性能优化建议

### 1. 批次大小调优

```json
{
  "datasets": {
    "train": {
      "dataloader_batch_size": 4  // 每卡批次，根据显存调整
    }
  }
}
```

**建议**：
- A100 80GB: batch_size = 8-16
- V100 32GB: batch_size = 4-8
- RTX 3090 24GB: batch_size = 2-4

### 2. 数据加载优化

```json
{
  "datasets": {
    "train": {
      "dataloader_num_workers": 8  // CPU 核心数的一半
    }
  }
}
```

### 3. 混合精度训练

在 `utils_option.py` 中启用：

```python
# 使用 AMP (Automatic Mixed Precision)
scaler = torch.cuda.amp.GradScaler()
```

可节省显存并加速训练（约 1.5-2 倍）。

### 4. 梯度累积

如果显存不足：

```python
# 每 4 步累积梯度后更新一次
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 调试技巧

### 1. 启用详细日志

```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
./launch_train.sh 4
```

### 2. 单卡测试

先用单卡确认代码正确：

```bash
python main_train_vrt.py --opt config.json
```

确认无误后再用多卡。

### 3. 小规模测试

修改配置用少量数据测试：

```json
{
  "datasets": {
    "train": {
      "dataloader_batch_size": 1
    }
  },
  "train": {
    "total_iter": 100  // 只训练 100 步
  }
}
```

### 4. 检查同步点

在关键位置添加同步：

```python
from utils.utils_dist import barrier

# 确保所有进程到达此处
barrier()
print(f"Rank {rank} passed checkpoint")
```

---

## 参考资料

### PyTorch 官方文档

- [分布式训练教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [DistributedDataParallel API](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [torchrun 使用指南](https://pytorch.org/docs/stable/elastic/run.html)

### NCCL 文档

- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL 性能调优](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/tuning.html)

---

## 变更历史

### v2.0 (2025-11-06)

- ✅ 完全重写分布式训练支持
- ✅ 支持 `env://` 初始化方式
- ✅ 自动检测平台 DDP 和本地训练
- ✅ 添加 `DistributedSampler` 支持
- ✅ 修复 `LOCAL_RANK` 设备分配问题
- ✅ 兼容 SLURM 集群环境
- ✅ 添加实用工具函数（`is_main_process()`, `barrier()` 等）

### v1.0 (Legacy)

- ⚠️ 旧版使用 `torch.distributed.launch`（已弃用）
- ⚠️ 手动设置 `CUDA_VISIBLE_DEVICES`（不兼容 DDP）
- ⚠️ 缺少 `DistributedSampler`（数据未正确分片）

---

## 联系与支持

如遇问题，请：

1. 检查日志中的错误信息
2. 启用 `NCCL_DEBUG=INFO` 查看详细信息
3. 参考本文档的"常见问题"部分
4. 在项目 GitHub 提交 Issue

---

**祝训练顺利！🚀**

