# Quick Reference Guide

**VRT+Spike Video Deblurring - Command Quick Reference**

> 💡 **新手？** 请先阅读 [快速开始指南](QUICK_START.md)  
> 📖 **详细配置？** 请查看 [配置指南](CONFIG_GUIDE.md)

---

## 🧪 Testing Commands

### Before Training
```bash
# System readiness check
python tests/integration/training/test_system_readiness.py

# Data pipeline test
python tests/integration/pipeline/test_training_dataloader.py

# Voxel generation test
python tests/unit/data/test_spike_voxel_realtime.py
```

### Run All Tests
```bash
# With pytest
pytest tests/ -v

# With custom runner
python tests/pytest_runner.py all
```

### Run Specific Categories
```bash
pytest tests/unit/ -v              # Unit tests
pytest tests/integration/ -v       # Integration tests
pytest tests/benchmark/ -v         # Benchmarks
```

### Performance Benchmark
```bash
python tests/benchmark/bench_data_loading.py
```

## 🔍 Common Tasks

### Check System Status
```bash
python tests/integration/training/test_system_readiness.py
```

### Test Data Loading
```bash
python tests/integration/pipeline/test_training_dataloader.py
```

### Benchmark Performance
```bash
python tests/benchmark/bench_data_loading.py
```

### Run All Tests
```bash
pytest tests/ -v
```

### Start Training
```bash
python src/train.py --config configs/deblur/vrt_spike_baseline.yaml
```

### Monitor Training
```bash
tensorboard --logdir outputs/logs/tb
```

---

## 📚 相关文档

- **[快速开始指南](QUICK_START.md)** - 新手入门
- **[配置指南](CONFIG_GUIDE.md)** - 完整配置说明
- **[训练问题排查](TRAINING_ISSUES.md)** - 问题诊断

---

**Last Updated**: 2025-10-21

