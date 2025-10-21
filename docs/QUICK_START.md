# Quick Start - Optimized Training

## 🚀 Run Training (3 GPUs)

```bash
cd /home/mallm/henry/Deblur
bash scripts/launch_train.sh
```

**Expected performance**: ≥2.0 samples/s (5× faster than baseline)

---

## 📊 Run Profiling (1 GPU, 100 steps)

```bash
bash scripts/profile_train.sh
```

View results:
```bash
tensorboard --logdir=outputs/prof/
```

---

## 📈 Monitor Training

### Real-time Metrics
```bash
# Terminal 1: Training logs
tail -f outputs/logs/train_*.log

# Terminal 2: GPU monitoring
watch nvidia-smi

# Terminal 3: TensorBoard
tensorboard --logdir=outputs/logs/tb
```

### Key Metrics
- **samples/s**: Target ≥2.0 (shown in logs every 50 steps)
- **GPU util**: Target >90% (nvidia-smi)
- **data_time_ms**: Target <10 (shown in logs)
- **loss**: Should decrease smoothly (TensorBoard)

---

## ⚙️ Configuration

**File**: `configs/deblur/vrt_spike_baseline.yaml`

### Key Settings

```yaml
DATA:
  USE_RAM_CACHE: true  # ⚡ 2-3× speedup (needs ~15 GB RAM)

TRAIN:
  COMPILE_MODEL: true  # ⚡ 1.2× speedup (PyTorch 2.0+)
  BATCH_SIZE: 1
  GRADIENT_ACCUMULATION_STEPS: 12  # Effective batch = 36
  NUM_WORKERS: 36  # 12 per GPU
  PREFETCH_FACTOR: 4
```

---

## 🛠️ Troubleshooting

### Out of Memory?

**System RAM**:
```yaml
DATA:
  USE_RAM_CACHE: false  # Disable caching
```

**GPU VRAM**:
```yaml
TRAIN:
  NUM_WORKERS: 12  # Reduce from 36
```

### torch.compile errors?

```yaml
TRAIN:
  COMPILE_MODEL: false  # Disable compilation
```

### Low throughput?

1. Check data_time_ms in logs (should be <10)
2. Verify GPU util >90% with `nvidia-smi`
3. Run profiler: `bash scripts/profile_train.sh`

---

## 📚 Documentation

- **Full guide**: `docs/TRAINING_OPTIMIZATION_GUIDE.md`
- **Testing**: `docs/OPTIMIZATION_TESTING_CHECKLIST.md`
- **Summary**: `OPTIMIZATION_SUMMARY.md`

---

## 🎯 Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Throughput | 0.4/s | ≥2.0/s |
| GPU util | 60-70% | >90% |
| Data time | ~100ms | <10ms |
| Batch time | ~2.5s | <500ms |

---

## ✅ Quick Validation

```bash
# 1. Start training
bash scripts/launch_train.sh

# 2. Check logs after ~100 steps
tail outputs/logs/train_*.log

# 3. Look for:
# "2.3 samples/s | data_time=4.2ms"
#       ^^^              ^^^
#    Target ≥2.0      Target <10
```

**If both targets met**: ✅ Optimization successful!

---

## 🔧 Advanced

### Single GPU training:
```bash
# Edit scripts/launch_train.sh
NUM_GPUS=1
```

### Profile specific steps:
```bash
python src/train.py \
  --config configs/deblur/vrt_spike_baseline.yaml \
  --profile --profile_steps 200
```

### Resume from checkpoint:
```bash
# (Feature to be added if needed)
```

---

## 📚 Additional Resources

### Memory Optimization

如果遇到OOM（内存不足）错误，请参阅：
- **[MEMORY_OPTIMIZATION.md](MEMORY_OPTIMIZATION.md)** - 详细的内存优化指南
  - LRU缓存策略说明
  - Worker数量调优
  - 内存监控工具
  - 故障排除指南

**快速修复OOM：**
```yaml
# 编辑 configs/deblur/vrt_spike_baseline.yaml
DATA:
  CACHE_SIZE_GB: 30.0  # 降低缓存大小（默认50GB）

TRAIN:
  NUM_WORKERS: 8       # 减少workers（默认12）
```

### Documentation

- [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) - 完整优化实施总结
- [docs/](docs/) - 详细技术文档

---

**Ready to train!** 🚀

