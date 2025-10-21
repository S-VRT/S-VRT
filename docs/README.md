# VRT+Spike 文档中心

欢迎来到 VRT+Spike 视频去模糊项目的文档中心！本目录包含所有技术文档、架构说明和开发指南。

---

## 📚 文档导航

### 🚀 核心文档（推荐阅读）

如果您是新手，建议按以下顺序阅读：

1. **[快速开始指南](QUICK_START.md)** ⭐ **必读**（~10分钟）
   - 环境配置和训练启动
   - 基础监控命令
   - 新手友好的入门指南

2. **[架构完整指南](ARCHITECTURE.md)** ⭐ **核心参考**（~30分钟）
   - 快速参考（ASCII图表）
   - 高层次架构
   - 详细数据流
   - 各阶段维度变换
   - 模块接口规范

3. **[推理策略完整指南](INFERENCE_GUIDE.md)** ⭐ **新增**（~25分钟）
   - 训练vs验证的裁剪策略
   - Tile Inference详解
   - 其他推理方法对比
   - 实际应用建议

### 📖 配置与使用指南

4. **[配置指南](CONFIG_GUIDE.md)**（~30分钟）
   - 完整的配置参数说明
   - YAML配置文件详解

5. **[快速配置参考](QUICK_CONFIG_REFERENCE.md)** 🔥 **速查**（~3分钟）
   - 最常用配置项速查表
   - 性能预设方案
   - 配置检查清单

6. **[快速命令参考](QUICK_REFERENCE.md)** 🔥 **速查**（~2分钟）
   - 测试命令集合
   - 常用操作速查

7. **[数据指南](DATA_GUIDE.md)**（~20分钟）
   - 数据集格式要求
   - 数据准备步骤
   - Spike数据处理

### 🔧 训练与优化

8. **[训练恢复指南](RESUME_TRAINING.md)**（~10分钟）
   - 如何恢复中断的训练
   - 检查点管理

9. **[性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)**（~25分钟）
   - 性能优化策略
   - 内存优化技巧
   - 完整优化方案

10. **[耗时分析指南](TIMING_GUIDE.md)**（~20分钟）
    - 耗时分析报告
    - 耗时日志系统
    - 性能调试指南

### 🧪 调试与诊断

11. **[训练问题排查](TRAINING_ISSUES.md)**（~10分钟）
    - 常见训练问题
    - 解决方案
    - NaN值诊断

---

## 🗂️ 文档分类速查

### 📖 按类型分类

| 类型 | 文档列表 | 总数 |
|------|---------|------|
| **🚀 快速入门** | QUICK_START, QUICK_REFERENCE, QUICK_CONFIG_REFERENCE | 3个 |
| **🏗️ 架构与设计** | ARCHITECTURE | 1个 |
| **⚙️ 配置指南** | CONFIG_GUIDE, DATA_GUIDE | 2个 |
| **🔧 训练与优化** | RESUME_TRAINING, PERFORMANCE_OPTIMIZATION_GUIDE, TIMING_GUIDE | 3个 |
| **🧪 推理与测试** | INFERENCE_GUIDE | 1个 |
| **🐛 调试排查** | TRAINING_ISSUES | 1个 |

**核心文档总数**: 11个（精简后）

---

## 🎯 按任务查找文档

### "我是新手，想快速入门"

👉 开始阅读：
1. [快速开始指南](QUICK_START.md) - 环境配置和首次运行
2. [架构完整指南](ARCHITECTURE.md) - 理解模型架构
3. [快速配置参考](QUICK_CONFIG_REFERENCE.md) - 常用配置速查

### "我想理解整体架构"

👉 开始阅读：
1. [架构完整指南](ARCHITECTURE.md) - 完整的架构说明
2. [验证与实现综合报告](VRT_Spike_验证与实现综合报告.md) - 验证实现细节
3. [Spike数据读取流程分析](Spike数据读取流程分析.md) - 理解数据处理

### "我想配置和训练模型"

👉 开始阅读：
1. [快速开始指南](QUICK_START.md) - 基础训练流程
2. [配置指南](CONFIG_GUIDE.md) - 详细配置说明
3. [快速配置参考](QUICK_CONFIG_REFERENCE.md) - 快速调整配置
4. [训练恢复指南](RESUME_TRAINING.md) - 恢复中断的训练

### "我遇到了性能或内存问题"

👉 开始阅读：
1. **[性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)** - 完整的性能和内存优化
2. [耗时分析指南](TIMING_GUIDE.md) - 性能瓶颈分析
3. [数据指南](DATA_GUIDE.md) - 数据加载优化

### "我遇到了训练错误"

👉 开始阅读：
1. **[训练问题排查](TRAINING_ISSUES.md)** - 常见问题和解决方案
2. [性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md) - 内存问题诊断
3. [耗时分析指南](TIMING_GUIDE.md) - 性能问题诊断

### "我想进行推理/测试"

👉 开始阅读：
1. **[推理策略完整指南](INFERENCE_GUIDE.md)** - 完整的推理方法说明
2. [快速命令参考](QUICK_REFERENCE.md) - 推理命令速查

### "我想修改或扩展代码"

👉 开始阅读：
1. **[架构完整指南](ARCHITECTURE.md)** - 理解数据流和接口
2. [配置指南](CONFIG_GUIDE.md) - 理解配置系统

### "我想处理 Spike 数据"

👉 开始阅读：
1. **[数据指南](DATA_GUIDE.md)** - 数据准备和Spike数据处理
2. [架构完整指南](ARCHITECTURE.md) - Spike编码器章节

---

## 📊 核心概念速查

### 关键组件

| 组件 | 功能 | 输入 | 输出 | 详细文档 |
|------|------|------|------|---------|
| **VRTWithSpike** | 主模型 | RGB+Spike | 清晰帧 | [架构指南](ARCHITECTURE.md) |
| **SpikeEncoder3D** | Spike编码 | 体素化Spike | 多尺度特征 | [架构指南](ARCHITECTURE.md) |
| **SpikeTemporalSA** | 时序建模 | Spike特征 | 时序增强特征 | [架构指南](ARCHITECTURE.md) |
| **CrossAttnFuse** | 跨模态融合 | RGB+Spike特征 | 融合特征 | [架构指南](ARCHITECTURE.md) |

### 关键维度

| 符號 | 含义 | 典型值 |
|------|------|--------|
| B | Batch size | 2-8 |
| T | 时间帧数 | 3-7 |
| K | Spike bins | 32 |
| H, W | 空间分辨率 | 128-384 |
| C | 特征通道数 | 64-96 |

更多详情请参考：[架构指南 - 维度速查表](ARCHITECTURE.md)

### 多尺度特征

```
Scale 1: 原始分辨率    (H × W)
Scale 2: 1/2 分辨率    (H/2 × W/2)
Scale 3: 1/4 分辨率    (H/4 × W/4)
Scale 4: 1/8 分辨率    (H/8 × W/8)
```

每个尺度都有对应的：
- **Fr**: VRT RGB 特征
- **Fs**: Spike 编码特征
- **Fs'**: 时序增强 Spike 特征
- **Ff**: 融合特征

详细说明：[架构指南 - 多尺度融合](ARCHITECTURE.md)

---

## 🛠️ 常用配置模板

### 标准训练配置

```yaml
# configs/deblur/vrt_spike_baseline.yaml
DATA:
  CLIP_LEN: 5
  CROP_SIZE: 256
  K: 32

MODEL:
  USE_SPIKE: true
  CHANNELS_PER_SCALE: [96, 96, 96, 96]
  
TRAIN:
  BATCH_SIZE: 2
  MIXED_PRECISION: true
```

详细配置选项：[配置指南](CONFIG_GUIDE.md) | [快速配置参考](QUICK_CONFIG_REFERENCE.md)

### 低显存配置

```yaml
# 适用于 <8GB 显存
DATA:
  CLIP_LEN: 3
  CROP_SIZE: 128

MODEL:
  VRT:
    USE_CHECKPOINT_ATTN: true
    USE_CHECKPOINT_FFN: true
  SPIKE_TSA:
    ADAPTIVE_CHUNK: true
  FUSE:
    ADAPTIVE_CHUNK: true

TRAIN:
  BATCH_SIZE: 1
  ACCUMULATE_GRAD: 4
```

更多场景：[架构快速参考 - 常见配置场景](architecture_quick_reference.md#-常见配置场景)

---

## 🔗 外部资源

### 相关论文

1. **VRT**: Video Restoration Transformer
   - Paper: [arXiv](https://arxiv.org/abs/2201.12288)
   - Code: [GitHub](https://github.com/JingyunLiang/VRT)

2. **Spike Camera**: 高时间分辨率视觉傳感器
   - Survey: Spike Camera and Its Applications (待补充链接)

3. **Video Deblurring**: 视频去模糊综述
   - Survey: Deep Video Deblurring (待补充链接)

### 相关代碼库

- **VRT 原始实现**: [third_party/VRT/](../third_party/VRT/)
- **SpikeCV**: [third_party/SpikeCV/](../third_party/SpikeCV/)

---

## 📝 文档維护

### 文档版本

所有文档当前版本：**v2.0** (2025-10-20)

### 贡献指南

如果您想改进文档，请：

1. 确保新增内容与现有文档风格一致
2. 更新相关的交叉引用
3. 在文档底部注明变更日期和版本
4. 更新本 README 中的目录

### 报告问题

如果您发现文档中的错误或不清晰之处，请：

1. 在项目 issue tracker 中报告
2. 清楚说明问题所在的文档和章节
3. 如可能，建议改进方案

---

## 🌟 推荐学习路径

### 初学者路径（2小时）

```
第1步 (20分钟) → 快速开始指南
                 ├─ 环境配置
                 ├─ 数据准备
                 └─ 首次训练

第2步 (30分钟) → 架构完整指南（快速参考部分）
                 ├─ 快速概览
                 ├─ 维度速查表
                 └─ 关键配置参数

第3步 (30分钟) → 动手实验
                 ├─ 运行小规模训练
                 ├─ 查看训练日志
                 └─ 对照文档理解

第4步 (40分钟) → 架构完整指南（详细部分）
                 ├─ 详细数据流
                 ├─ 各阶段维度变换
                 └─ 模块接口规范
```

### 开发者路径（4小时）

```
第1步 (30分钟) → 快速开始
                 ├─ 快速开始指南
                 ├─ 架构完整指南（快速参考）
                 └─ 快速配置参考

第2步 (90分钟) → 代码探索
                 ├─ src/models/integrate_vrt.py
                 ├─ src/models/spike_encoder3d.py
                 ├─ src/models/spike_temporal_sa.py
                 └─ src/models/fusion/cross_attn_temporal.py
                 └─ 对照架构完整指南理解

第3步 (60分钟) → 深入理解
                 ├─ 验证与实现综合报告（了解实现状态）
                 ├─ 配置指南（了解配置系统）
                 └─ Spike数据读取流程（了解数据处理）

第4步 (60分钟) → 实践训练
                 ├─ 配置训练环境
                 ├─ 运行完整训练
                 └─ 根据优化指南调优
```

### 研究者路径（全天）

```
上午 (3小时) → 完整技术理解
       ├─ 架构完整指南（全文）
       ├─ 验证与实现综合报告
       ├─ 代码全面阅读
       └─ 验证推理策略详解

下午 (4小时) → 深度探索
       ├─ 配置不同实验
       ├─ 性能分析（时序分析指南）
       ├─ 参数敏感性实验
       └─ 内存/性能优化

晚上 (2小时) → 扩展研究
       ├─ 改进方案设计
       ├─ 文档记录
       └─ 实验计划
```

---

## ❓ FAQ

### Q1: 我应该从哪个文档开始？

**A**: 推荐从 [快速开始指南](QUICK_START.md) 开始，然后阅读 [架构完整指南](ARCHITECTURE.md) 的快速参考部分。

### Q2: 我是新手，如何快速上手？

**A**: 按照以下步骤：
1. [快速开始指南](QUICK_START.md) - 配置环境
2. [快速配置参考](QUICK_CONFIG_REFERENCE.md) - 了解常用配置
3. [快速命令参考](QUICK_REFERENCE.md) - 学习常用命令
4. 运行小规模训练实验

### Q3: 配置参数很多，有没有推荐设置？

**A**: 有的，查看：
- [快速配置参考](QUICK_CONFIG_REFERENCE.md) - 最常用配置
- [配置指南](CONFIG_GUIDE.md) - 详细配置说明
- [架构完整指南](ARCHITECTURE.md) 的配置参数章节

### Q4: 如何优化显存使用？

**A**: 参考以下资源：
- **[性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md)** - 完整的性能和内存优化
- [快速配置参考](QUICK_CONFIG_REFERENCE.md) - 低显存配置
- [耗时分析指南](TIMING_GUIDE.md) - 性能瓶颈分析

### Q5: Spike 数据格式是什么？

**A**: 详细说明请查看：
- **[数据指南](DATA_GUIDE.md)** - 完整的数据准备和Spike数据处理

### Q6: 我遇到训练错误，如何排查？

**A**: 
1. 查看 **[训练问题排查](TRAINING_ISSUES.md)** - 常见问题和解决方案
2. 参考 [性能优化指南](PERFORMANCE_OPTIMIZATION_GUIDE.md) - 内存问题诊断
3. 使用 [耗时分析指南](TIMING_GUIDE.md) - 性能瓶颈分析

### Q7: 如何理解模型架构？

**A**: 建议阅读顺序：
1. [架构完整指南](ARCHITECTURE.md) - 快速参考部分（10分钟）
2. [架构完整指南](ARCHITECTURE.md) - 详细数据流（20分钟）
3. [验证与实现综合报告](VRT_Spike_验证与实现综合报告.md) - 实现细节

### Q8: 如何进行大图像推理？

**A**: 查看 **[推理策略完整指南](INFERENCE_GUIDE.md)** - 包含Tile推理的完整说明。

### Q9: 训练中断了如何恢复？

**A**: 参考 [训练恢复指南](RESUME_TRAINING.md)，里面有详细的恢复步骤。

### Q10: 在哪里可以找到历史开发文档？

**A**: 所有历史文档都已归档在 `archive/` 目录中，包括早期的设计文档和核验报告。

---

## 📧 聯繫与支持

- **项目主頁**: (待补充)
- **Issue Tracker**: (待补充)
- **讨论区**: (待补充)

---

## 📄 許可证

本文档与代碼遵循项目的許可证条款。

---

## 📝 文档更新记录

### v4.1 (2025-10-21) - 深度文档整合 🎯
- ✅ 整合推理策略：归档 `验证推理策略详解.md`，简化验证报告中的重复内容
- ✅ 清理元文档：归档临时工作文档（3个分析报告文档）
- ✅ 更新所有交叉引用：移除对归档文档的引用
- ✅ 优化README导航：更新FAQ和速查表
- ✅ **核心文档: 17个 → 13个** (减少23%)
- ✅ **消除内容重复**: 推理策略统一到INFERENCE_GUIDE.md

### v4.0 (2025-10-21) - 推理指南整合与精简优化
- ✅ 新增 **[推理策略完整指南](INFERENCE_GUIDE.md)** (545行)
- ✅ 整合tile inference相关文档
- ✅ 归档旧版 `tile_inference_usage.md`

### v3.1 (2025-10-21) - 深度精简
- ✅ 整合3份时序文档 → `TIMING_GUIDE.md`
- ✅ 归档详细Spike数据分析（1051行）
- ✅ 移动示例代码到 `examples/` 目录
- ✅ 核心文档: 26个 → **18个** (减少30%)

### v3.0 (2025-10-21) - 整合综合指南
- ✅ 新增 [架构完整指南](ARCHITECTURE.md)
- ✅ 将历史文档归档到 `archive/` 目录
- ✅ 更新文档索引和导航

---

**文档中心版本**: v4.1  
**最后更新**: 2025-10-21  
**维护者**: VRT+Spike Development Team

💡 **提示**: 归档文档可在 [`archive/`](archive/) 目录查看  

---

<div align="center">
  
### 🎉 祝您学习愉快！

如有任何疑问，请先查阅相关文档，或在 issue tracker 中提问。

**Happy Coding! 🚀**

</div>
