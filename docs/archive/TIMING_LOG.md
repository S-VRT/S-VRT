# 训练耗时日志系统

## 概述

为了更好地分析训练过程中各个模块的耗时情况，我们实现了一个独立的耗时日志系统 (`TimingLogger`)。该系统提供了：

1. **原地更新的终端显示** - 避免终端日志刷屏
2. **详细的文件日志** - 记录每个step的完整耗时分布
3. **层次化的统计** - 支持嵌套模块的耗时记录
4. **实时统计** - 显示平均值、最小值、最大值

## 特性

### 1. 终端显示

终端显示采用原地更新，展示top 5耗时模块的进度条和百分比：

```
┌─ Timing Profile (Step 42) ─────────────────────
│ 前向传播总耗时                   █████████████░░░░░░░░░░░░░░░░░  280.0ms 45.2%
│ VRT处理                     ███████░░░░░░░░░░░░░░░░░░░░░░░  150.0ms 24.2%
│ Spike时间自注意力               ███░░░░░░░░░░░░░░░░░░░░░░░░░░░   80.0ms 12.9%
│ VRT融合                     ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   60.0ms  9.7%
│ Spike编码器                  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░   50.0ms  8.1%
└─ Total:  620.0ms ─────────────────────────────────────
```

### 2. 文件日志

详细日志会保存到文件中，格式如下：

```
================================================================================
Step 1
================================================================================
前向传播总耗时                                 :   280.00ms ( 45.2%) [avg:   280.00ms]
VRT处理                                   :   150.00ms ( 24.2%) [avg:   150.00ms]
  - Stage1                              :    25.00ms [avg:    25.00ms]
  - Stage2                              :    23.00ms [avg:    23.00ms]
  - Stage3                              :    22.00ms [avg:    22.00ms]
  ...
Spike时间自注意力                             :    80.00ms ( 12.9%) [avg:    80.00ms]
  - Self-Attention                      :    50.00ms [avg:    50.00ms]
  - FFN                                 :    20.00ms [avg:    20.00ms]
  - 维度转换                                :    10.00ms [avg:    10.00ms]
...
Total-----------------------------------:   620.00ms
```

### 3. 训练结束总结

训练结束时会打印完整的统计总结：

```
================================================================================
Timing Summary (Total 1000 steps)
================================================================================
前向传播总耗时                                 : avg= 280.00ms  min= 275.00ms  max= 290.00ms  (n=1000)
VRT处理                                   : avg= 150.00ms  min= 145.00ms  max= 160.00ms  (n=1000)
Spike时间自注意力                             : avg=  80.00ms  min=  78.00ms  max=  85.00ms  (n=1000)
...
```

## 使用方法

### 在训练代码中使用

```python
from src.utils.timing_logger import TimingLogger, set_global_timing_logger, log_timing

# 初始化logger (在main进程中)
timing_logger = TimingLogger(
    log_dir=save_root / "logs",
    enable_console=True,
    enable_file=True,
    console_update_interval=10,  # 每10个step更新一次终端显示
    file_flush_interval=50,      # 每50个step刷新一次文件
)
set_global_timing_logger(timing_logger)

# 在需要记录耗时的地方
log_timing("模块名称", time_in_ms)
log_timing("模块名称/子模块", time_in_ms)  # 支持层次化

# 每个训练step结束时
timing_logger.step()

# 训练结束时
timing_logger.print_summary()
timing_logger.close()
```

### 在模型代码中使用

```python
from src.utils.timing_logger import log_timing
import torch

class MyModule(nn.Module):
    def forward(self, x):
        # 方法1: 手动计时
        start = time.time()
        result = self.process(x)
        log_timing("MyModule/process", (time.time() - start) * 1000)
        
        # 方法2: 使用CUDA事件（更准确）
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result = self.process(x)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            log_timing("MyModule/process", elapsed_ms)
        
        return result
```

## 配置项

在 `config.yaml` 中可以配置以下选项：

```yaml
LOG:
  # 是否启用耗时日志
  ENABLE_TIMING_LOG: true
  
  # 是否在终端显示
  TIMING_CONSOLE: true
  
  # 是否记录到文件
  TIMING_FILE: true
  
  # 终端更新间隔（每N个step）
  TIMING_CONSOLE_INTERVAL: 10
  
  # 文件刷新间隔（每N个step）
  TIMING_FILE_INTERVAL: 50
```

## 实现细节

### 层次化支持

使用 `/` 分隔符来表示层次关系：

- `"VRT处理"` - 顶层模块
- `"VRT处理/Stage1"` - VRT处理的子模块
- `"VRT处理/Stage1/Attention"` - Stage1的子模块

文件日志中会使用缩进显示层次关系，终端只显示顶层模块。

### 性能考虑

1. **低开销** - 只在需要时才更新显示和写入文件
2. **异步友好** - 终端更新使用原地覆盖，不阻塞训练
3. **分布式训练** - 只在主进程中记录，避免冲突
4. **条件编译** - 可以通过配置完全关闭

### 线程安全

`TimingLogger` 使用了线程锁来保证并发安全，可以在多线程环境下使用。

## 输出示例

### 日志文件

日志文件保存在 `{SAVE_DIR}/logs/timing_{timestamp}.log`，例如：
- `outputs/run_001/logs/timing_1760698920.log`

### 文件格式

每个step占用一个独立的section，包含：
1. Step编号
2. 各模块耗时（按耗时降序排列）
3. 子模块耗时（缩进显示）
4. 百分比和累积平均值
5. 总耗时

## 最佳实践

1. **合理的命名** - 使用清晰的中英文模块名
2. **适当的粒度** - 记录关键模块，避免过于细粒度
3. **使用CUDA事件** - GPU操作使用CUDA事件计时更准确
4. **配置更新间隔** - 根据训练速度调整显示间隔
5. **保留日志文件** - 便于后续分析和对比

## 故障排除

### 终端显示乱码

确保终端支持ANSI转义序列和UTF-8编码。

### 文件未生成

检查：
1. `ENABLE_TIMING_LOG` 是否为 `true`
2. `TIMING_FILE` 是否为 `true`
3. 日志目录是否有写权限
4. 是否在主进程中运行

### 耗时不准确

对于GPU操作：
1. 使用 CUDA 事件而不是 `time.time()`
2. 确保在记录前调用 `torch.cuda.synchronize()`

### 性能影响

如果担心性能影响：
1. 增大 `console_update_interval` 和 `file_flush_interval`
2. 设置 `TIMING_CONSOLE: false` 只记录到文件
3. 设置 `ENABLE_TIMING_LOG: false` 完全关闭

## 未来改进

- [ ] 支持导出为JSON/CSV格式
- [ ] 支持可视化分析工具
- [ ] 支持自动检测性能瓶颈
- [ ] 支持与TensorBoard集成
- [ ] 支持分布式训练的多进程汇总

