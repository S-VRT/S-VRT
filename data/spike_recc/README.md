# Spike Loader Module

这个模块专门用于 Spike 数据的加载和重建。

## 目录结构

```
spike_recc/
├── __init__.py          # 模块导出
├── spikecv_loader.py      # SpikeCV 加载与重建方法（当前实现）
└── README.md            # 本文件
```

## 当前实现

### `spike_recc.py` (原 `spike_loader.py`)
- **SpikeStreamSimple**: 轻量封装 SpikeCV 的 `SpikeStream`，用于加载 .dat 文件
- **voxelize_spikes_tfp**: 使用 TFP (Time-Frequency Projection) 方法将 spike 流重建为图像
- **load_spike_dat**: 基础 spike 数据加载器
- **load_spike_dat_alternative**: 替代格式的 spike 数据加载器

## 使用示例

```python
from data.spike_recc import SpikeStreamSimple, voxelize_spikes_tfp

# 加载 spike 数据
spike_stream = SpikeStreamSimple('path/to/spike.dat', spike_h=250, spike_w=400)
spike_matrix = spike_stream.get_spike_matrix(flipud=True)

# 使用 TFP 重建
spike_voxel = voxelize_spikes_tfp(
    spike_matrix, 
    num_channels=1, 
    device='cpu', 
    half_win_length=20
)
```

## 扩展

未来可以在此文件夹中添加其他重建方法，例如：
- `spike_loader_snn.py`: SNN 重建方法
- `spike_loader_cnn.py`: CNN 重建方法
- `spike_loader_hybrid.py`: 混合重建方法
- 等等...

每个新的加载器/重建方法应该：
1. 提供清晰的接口
2. 在 `__init__.py` 中导出
3. 添加相应的文档说明
