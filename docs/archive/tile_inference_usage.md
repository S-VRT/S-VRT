# Tile Inference 功能使用说明

## 概述

`tile_inference` 函数实现了对大尺寸图像的分块推理，避免内存溢出（OOM）问题。该函数支持自动分块、手动控制分块行为，以及多种配置选项。

## 功能特性

### 1. 自动分块决策
- 当图像尺寸超过 `tile_size` 时自动启用分块
- 小图像直接处理，避免不必要的分块开销
- 支持强制启用或禁用分块

### 2. 平滑融合
- 使用余弦窗函数实现tile边界的平滑融合
- 避免tile边界处的伪影
- 支持自定义重叠区域大小

### 3. 错误处理
- 完整的输入验证
- 边界情况处理
- 数值稳定性检查
- 进度显示和错误报告

### 4. 性能优化
- 内存高效的tensor操作
- 支持GPU加速
- 进度跟踪和性能监控

## 使用方法

### 基本用法

```python
from src.test import tile_inference

# 基本调用
result = tile_inference(
    model=your_model,
    blur=blur_tensor,      # (B, T, C, H, W)
    spike_vox=spike_tensor, # (B, T, K, H_s, W_s)
    tile_size=256,
    tile_overlap=32,
    device=device
)
```

### 命令行参数

```bash
# 使用默认参数
python src/test.py --config config.yaml

# 自定义tile参数
python src/test.py --config config.yaml --tile_size 512 --tile_overlap 64

# 强制启用分块（即使对小图像）
python src/test.py --config config.yaml --enable_tiling

# 禁用分块（处理完整图像）
python src/test.py --config config.yaml --disable_tiling
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tile_size` | int | 256 | 每个tile的尺寸 |
| `tile_overlap` | int | 32 | tile之间的重叠区域 |
| `enable_tiling` | flag | False | 强制启用分块 |
| `disable_tiling` | flag | False | 禁用分块 |

## 配置建议

### 内存配置
- **GPU内存 < 8GB**: `tile_size=128, tile_overlap=16`
- **GPU内存 8-16GB**: `tile_size=256, tile_overlap=32`
- **GPU内存 > 16GB**: `tile_size=512, tile_overlap=64`

### 质量配置
- **高质量**: `tile_overlap = tile_size // 8`
- **平衡**: `tile_overlap = tile_size // 8`
- **快速**: `tile_overlap = tile_size // 16`

## 测试结果

所有测试用例均已通过：

✅ 小图像直接处理  
✅ 大图像自动分块  
✅ 强制分块模式  
✅ 禁用分块模式  
✅ 边界情况处理  
✅ 错误恢复机制  

## 注意事项

1. **内存使用**: 分块会减少峰值内存使用，但会增加总计算时间
2. **质量影响**: 适当的重叠区域可以保持图像质量
3. **设备兼容**: 支持CPU和GPU推理
4. **数值稳定**: 自动处理NaN和Inf值

## 故障排除

### 常见问题

1. **OOM错误**: 减小 `tile_size` 或增加 `tile_overlap`
2. **质量下降**: 增加 `tile_overlap` 或使用更大的 `tile_size`
3. **速度慢**: 增大 `tile_size` 或减小 `tile_overlap`

### 调试信息

函数会输出详细的调试信息：
- 模型配置信息
- 分块决策过程
- 处理进度
- 警告和错误信息



