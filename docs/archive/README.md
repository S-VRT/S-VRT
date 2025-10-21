# 文档归档

本目录包含已被整合或过时的历史文档，保留作为参考。

## 📂 归档分类

### 架构文档（已整合到 ARCHITECTURE.md）

- `architecture_dataflow.md` - 数据流详细分析
- `architecture_diagrams.md` - 架构可视化图表
- `architecture_quick_reference.md` - 架构速查表

**新文档**: `../ARCHITECTURE.md` - 整合了以上所有内容

### 验证报告（已整合到 VRT_Spike_验证与实现综合报告.md）

- `CODE_VERIFICATION_REPORT.md` - 代码验证报告
- `核验.md` - 核验记录
- `模块核验总结.md` - 模块核验总结
- `模块核验报告.md` - 详细核验报告  
- `项目实现现状详细核验.md` - 实现现状核验

**新文档**: `../VRT_Spike_验证与实现综合报告.md` - 整合了以上所有验证报告

### 时序分析文档（已整合到 TIMING_GUIDE.md）

- `TIMING_ANALYSIS_REVELATION.md` - 耗时分析报告
- `TIMING_LOG.md` - 日志系统说明
- `TIMING_DEBUG_GUIDE.md` - 调试指南

**新文档**: `../TIMING_GUIDE.md` - 整合了所有时序相关文档

### 历史开发文档

- `VRT+Spike Baseline 实施进度.md` (2025-10-09)
- `VRT+Spike 视频去模糊 Baseline 开发指导.md` (2025-10-09)
- `VRT+Spike 融合视频去模糊.md` (2025-09-16)
- `架构迁移总结.md` (2025-10-09)

这些是项目早期开发过程中的计划和指导文档。

### 参考文档

- `Spike数据读取流程分析.md` - 非常详细的Spike数据处理流程（1051行）
  - 过于详细，日常开发参考 `../DATALOADER_GUIDE.md` 即可
  - 需要深入了解SpikeCV实现时可参考此文档

## 📚 使用说明

1. **日常开发** - 请使用 `docs/` 根目录下的新文档
2. **深入研究** - 需要详细历史信息时可查看此目录
3. **不建议修改** - 这些是历史文档，不应再修改

## 🔄 归档历史

### 2025-10-21 - 文档整合
- 整合3份架构文档 → `ARCHITECTURE.md`
- 整合5份验证报告 → `VRT_Spike_验证与实现综合报告.md`
- 整合3份时序文档 → `TIMING_GUIDE.md`
- 归档详细的Spike分析文档
- 清理临时组织文档

### 2025-10-20 - 初始归档
- 移入早期开发文档
- 保留历史核验报告

---

**维护者**: Deblur项目团队  
**最后更新**: 2025-10-21
