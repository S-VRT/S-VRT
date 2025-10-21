# 快速配置参考

> 💡 **完整配置说明？** 请查看 [配置指南](CONFIG_GUIDE.md)  
> 🚀 **快速入门？** 请查看 [快速开始指南](QUICK_START.md)

本文档提供最常用配置的速查表。

---

## 🚀 最常用的配置项

### 数据加载性能

```yaml
DATALOADER:
  TOTAL_WORKERS: "auto"          # 推荐！自动配置workers
  TRAIN_PREFETCH_FACTOR: 4       # 预取4个批次
  PIN_MEMORY: true               # 必须开启
  PERSISTENT_WORKERS: true       # 必须开启
```

**其他选项**：
- `TOTAL_WORKERS: "cpu*0.9"` - 使用90%的CPU
- `TOTAL_WORKERS: 24` - 固定24个workers

---

### 耗时日志（Timing Logger）

```yaml
LOG:
  SAVE_DIR: outputs              # ⚠️ 必须设置！
  ENABLE_TIMING_LOG: true        # 启用耗时日志
  TIMING_CONSOLE: true           # 终端显示进度条
  TIMING_FILE: true              # 保存详细日志
  TIMING_CONSOLE_INTERVAL: 10    # 每10步更新
  TIMING_FILE_INTERVAL: 50       # 每50步写入
```

**日志输出位置**：`{SAVE_DIR}/run_xxx/logs/timing_*.log`

---

### 训练参数

```yaml
TRAIN:
  BATCH_SIZE: 1                  # 每GPU的batch size
  GRADIENT_ACCUMULATION_STEPS: 6 # 梯度累积（等效batch=6）
  EPOCHS: 40
```

**有效batch size** = `BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS × GPU数量`

例如：`1 × 6 × 3 = 18`

---

## ⚠️ 常见错误

### 错误1: LOG部分重复定义

❌ **错误**：
```yaml
LOG:
  SAVE_DIR: outputs

LOG:  # 重复！会覆盖上面的
  ENABLE_TIMING_LOG: true
```

✅ **正确**：
```yaml
LOG:
  SAVE_DIR: outputs
  ENABLE_TIMING_LOG: true
```

### 错误2: 缺少 SAVE_DIR

```
KeyError: 'SAVE_DIR'
```

**解决**：在 `LOG` 部分添加 `SAVE_DIR: outputs`

---

## 📊 性能预设

### 高性能训练（3×A6000, 40核）

```yaml
DATALOADER:
  TOTAL_WORKERS: "auto"
  TRAIN_PREFETCH_FACTOR: 4
  PERSISTENT_WORKERS: true

TRAIN:
  BATCH_SIZE: 2                  # 如果显存够
  GRADIENT_ACCUMULATION_STEPS: 3

DATA:
  USE_RAM_CACHE: true
  CACHE_SIZE_GB: 4.0             # 每GPU 4GB缓存
```

### 调试模式

```yaml
DATALOADER:
  TOTAL_WORKERS: 4               # 固定少量workers
  TRAIN_PREFETCH_FACTOR: 2

TRAIN:
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 1

LOG:
  TIMING_CONSOLE_INTERVAL: 1     # 每步显示
```

### 显存有限

```yaml
DATALOADER:
  TOTAL_WORKERS: "cpu*0.5"       # 减少workers
  TRAIN_PREFETCH_FACTOR: 2

TRAIN:
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 12

DATA:
  USE_RAM_CACHE: false           # 关闭缓存
```

---

## 🔍 配置检查清单

启动训练前检查：

- [ ] `LOG.SAVE_DIR` 已设置
- [ ] `DATALOADER.TOTAL_WORKERS` 已配置
- [ ] `TRAIN.BATCH_SIZE` 适合显存
- [ ] `DATA.ROOT` 路径正确
- [ ] `MODEL.VRT_CFG` 文件存在

---


## 📚 相关文档

- **[配置指南](CONFIG_GUIDE.md)** - 完整配置参数说明
- **[快速开始指南](QUICK_START.md)** - 训练启动指南
- **[性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)** - 性能调优

---

**Last Updated**: 2025-10-21

