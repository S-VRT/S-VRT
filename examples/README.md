# Examples

本目录包含项目的示例代码和参考实现。

## 📁 文件说明

### SpikeCV示例代码

- `deblur_x4k_dataset.py` - X4K数据集的SpikeCV读取示例
- `deblur_read_x4k_spk_val_data.py` - X4K Spike验证数据读取示例
- `deblur_x4k_sequence.py` - X4K序列数据处理示例

这些文件展示了如何使用SpikeCV库读取和处理Spike相机数据。由于竞赛方给出了若干读取数据所使用的脚本，故将其放进项目内。

## 📚 相关文档

详细的数据加载和处理说明，请参考：
- [数据集接入与使用指南](../docs/数据集接入与使用指南.md)
- [DataLoader完整指南](../docs/DATALOADER_GUIDE.md)
- [Spike数据读取流程分析](../docs/archive/Spike数据读取流程分析.md) (归档)

## 🚀 使用方法

这些示例代码主要用于学习和参考，不是项目训练的必需部分。

项目实际使用的数据加载实现在：
- `src/data/datasets/spike_deblur_dataset.py`
- `src/data/vendors/` (供应商特定实现)

---

**注意**: 这些示例代码可能使用旧版API或不同的配置，仅供参考。


