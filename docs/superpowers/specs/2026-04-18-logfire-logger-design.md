# Logfire 接入统一 Logger 设计

**状态**：已设计，待实施
**范围**：当前仅覆盖主训练链路的统一日志管理层，不扩展到 `KAIR/`、`spike_stream_reconstruct/` 等独立日志入口。

## 1. 目标

将项目现有统一日志管理层接入 Logfire，作为一个新的可选后端，与现有 TensorBoard、W&B、SwanLab 并行输出。

本次设计目标是：
- 保持现有训练调用点基本不变。
- 让文本日志、结构化标量指标、timings 都能进入 Logfire。
- 在 Logfire 不可用、认证失败或网络异常时，不影响训练主流程。
- 保持与分布式训练当前行为一致，避免多 rank 重复上报。

## 2. 现状

项目当前已经存在一个统一日志管理入口：
- [main_train_vrt.py](../../../main_train_vrt.py) 在训练入口中初始化标准 `logging.Logger` 与统一 `utils_logger.Logger`。
- [utils/utils_logger.py](../../../utils/utils_logger.py) 中的 `Logger` 已统一管理文件日志、TensorBoard、W&B、SwanLab。
- [utils/utils_option.py](../../../utils/utils_option.py) 负责为 `opt['logging']` 提供默认配置项。

当前日志大致分成两层：
1. **文本日志**：由标准 `logging.Logger` 负责，输出到文件与终端。
2. **结构化实验日志**：由 `utils_logger.Logger` 负责，输出到 TensorBoard、W&B、SwanLab。

因此，若要把“本项目的 log 放进 Logfire”，需要同时覆盖这两层。

## 3. 候选方案

### 3.1 方案 A：只在统一 `Logger` 内增加 Logfire 后端

在 `utils_logger.Logger` 内仿照 W&B / SwanLab 增加 Logfire 初始化，并在 `log_scalars()` 与 `log_timings()` 中并行上报。

**优点**：
- 改动最小。
- 与现有统一管理结构一致。
- 不需要改训练循环调用点。

**缺点**：
- 会漏掉大量通过 `logger.info(...)` 发出的文本日志。

### 3.2 方案 B：只给标准 `logging` 增加 Logfire handler

在 `logger_info()` 中给标准 Python logger 增加一个可选的 Logfire handler，让现有文本日志自动进入 Logfire。

**优点**：
- 训练入口中的现有文本日志几乎无须改动即可纳管。

**缺点**：
- 无法覆盖 `log_scalars()` / `log_timings()` 这类结构化实验指标。

### 3.3 方案 C：混合方案

同时实现：
- 标准 `logging` → Logfire handler，用于文本日志。
- 统一 `Logger` → Logfire backend，用于 scalars 与 timings。

**优点**：
- 覆盖最完整。
- 最符合“统一管理”和“并行输出”的目标。
- 仍可把改动集中在少数文件中完成。

**缺点**：
- 实现复杂度高于单一方案，但仍然可控。

## 4. 设计选择

本设计采用 **方案 C（混合方案）**。

理由：
- 项目已经具备统一日志管理层，最合适的切入点就是 [utils/utils_logger.py](../../../utils/utils_logger.py)。
- 当前训练过程中的关键信息分散在文本日志与结构化指标两套路径中，只覆盖其中一层会导致 Logfire 视图不完整。
- 通过将改动限定在 `utils_option.py` 与 `utils_logger.py`，可以在不扰动训练主流程的前提下完成接入。

## 5. 架构设计

### 5.1 配置层

在 [utils/utils_option.py](../../../utils/utils_option.py) 的 `opt['logging']` 默认配置中新增以下字段：

- `use_logfire`: 是否启用 Logfire，默认 `false`
- `logfire_token`: Logfire 认证 token，默认 `null`
- `logfire_project_name`: 项目标识，默认 `null`
- `logfire_service_name`: 服务名，默认 `"s-vrt"`
- `logfire_environment`: 运行环境标识，默认 `null`
- `logfire_log_text`: 是否上报文本日志，默认 `true`
- `logfire_log_metrics`: 是否上报 scalar 指标，默认 `true`
- `logfire_log_timings`: 是否上报 timings，默认 `true`

配置保持放在现有 `options/*.json` 的 `logging` 段中，与 W&B / SwanLab 的使用方式一致。

### 5.2 Logfire 依赖初始化

在 [utils/utils_logger.py](../../../utils/utils_logger.py) 顶部增加 `logfire` 的可选导入：
- 导入成功则允许初始化。
- 导入失败则仅在启用 `use_logfire=true` 时输出 warning，并自动禁用 Logfire。

不引入强依赖。训练环境未安装 Logfire 时，训练应按原有行为继续运行。

### 5.3 统一 Logger 内的 Logfire backend

在 `Logger.__init__()` 中读取新增配置项，并在满足以下条件时初始化 Logfire：
- `use_logfire == true`
- `logfire` 包可导入
- 初始化未抛出异常

初始化后保存 Logfire 客户端或已配置状态到 `self` 上，供后续 `log_scalars()` / `log_timings()` 使用。

### 5.4 标准 logging 的 Logfire 桥接

在 `logger_info()` 中，当满足以下条件时，为标准 `logging.Logger` 挂载一个可选的 Logfire handler：
- 当前 rank 为 0
- `use_logfire == true`
- `logfire_log_text == true`
- 尚未挂载过对应 handler

这样现有的 `logger.info()` / `logger.warning()` 文本日志即可自动进入 Logfire，而无需逐个修改调用点。

### 5.5 分布式训练行为

Logfire 上报与当前文件日志策略保持一致：
- **默认仅 rank 0 上报**。
- 非主进程不挂载文本日志 handler，也不发送 metrics/timings。

这样可避免在 DDP 下重复记录多份相同事件。

## 6. 数据模型

### 6.1 文本日志

现有文本日志继续通过标准 `logging` 输出到文件和终端，同时并行发往 Logfire。

进入 Logfire 的日志应附带统一上下文字段：
- `task`
- `opt_path`
- `rank`
- `world_size`
- `is_train`
- `project_name`
- `run_name`

这些字段用于在 Logfire 中按实验与运行上下文进行过滤，而不改变现有本地日志格式。

### 6.2 标量指标

`log_scalars(step, scalar_dict, tag_prefix)` 继续沿用现有 tag 体系，并映射为结构化字段：
- `train/loss`
- `train/learning_rate`
- `test/psnr`
- `time/data`

每次上报至少附带：
- `step`
- 指标字典
- tag 前缀

如果调用侧未来提供 `epoch`，可以作为附加字段一起上报，但第一版不要求修改现有调用接口。

### 6.3 Timings

`log_timings()` 当前既会生成文本消息，也会向实验后端发送 timing 数值。接入 Logfire 后：
- 文本消息仍保留。
- timing 数值应作为结构化字段单独发送，而不是仅发送一段拼接字符串。

这样可以在 Logfire 中直接对 timing 指标聚合和筛选。

## 7. 失败与退化策略

Logfire 是一个可选后端，不能影响训练主流程。

### 7.1 初始化失败

以下情况都只应导致 Logfire 被禁用，不得中断训练：
- `logfire` 包未安装
- token 无效
- 配置不完整
- 远程初始化失败

处理方式：
- 输出一条 warning 到现有 logger 或标准输出
- 将内部 `use_logfire` 状态置为禁用
- 其余 TensorBoard / W&B / SwanLab 路径继续按原逻辑运行

### 7.2 运行期上报失败

文本日志、metrics、timings 三类上报都应各自独立 `try/except`。

要求：
- 某类上报失败不影响其他类型。
- 某类连续失败后，可在本进程内熔断关闭该类 Logfire 输出，避免每个 step 重复报错刷屏。
- 不向调用方抛出异常。

### 7.3 敏感信息处理

虽然当前设计允许把 `logfire_token` 放进 `options/*.json`，但不应把完整 `opt` 原样发送到 Logfire。

只发送筛选后的运行元数据，避免将 API key、token 或其他敏感配置二次上传。

## 8. 第一版落点与最小范围

第一版改动限定在以下文件：
- [utils/utils_option.py](../../../utils/utils_option.py)
- [utils/utils_logger.py](../../../utils/utils_logger.py)

不改动：
- [main_train_vrt.py](../../../main_train_vrt.py) 的训练循环逻辑
- `KAIR/` 下各类独立测试与训练脚本
- `spike_stream_reconstruct/` 下独立日志实现
- 图像上传到 Logfire

第一版只覆盖：
- 文本日志
- scalar 指标
- timings

## 9. 配置示例

```json
"logging": {
  "use_tensorboard": true,
  "use_wandb": true,
  "use_swanlab": false,
  "use_logfire": true,
  "logfire_token": "xxx",
  "logfire_project_name": "Deblur",
  "logfire_service_name": "s-vrt",
  "logfire_environment": "local",
  "logfire_log_text": true,
  "logfire_log_metrics": true,
  "logfire_log_timings": true
}
```

## 10. 测试要求

至少验证以下场景：

1. **未安装 Logfire**
   - `use_logfire=true` 时训练仍可启动。
   - 仅出现一次清晰 warning。
   - 其他日志后端照常工作。

2. **正常启用 Logfire**
   - 文本日志可进入 Logfire。
   - `train/*`、`test/*`、`time/*` 指标可进入 Logfire。
   - 本地文件日志与现有后端行为不受影响。

3. **分布式训练**
   - 仅 rank 0 上报一次。
   - 不出现每个 rank 都重复发送相同日志的问题。

4. **上报失败**
   - 网络失败或认证失败不会中断训练。
   - 失败后不会在每个 step 无限刷屏。

## 11. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 接入方式 | 混合方案 | 文本日志与结构化指标分属两条路径，需要同时覆盖 |
| 配置位置 | 继续放在 `logging` 段 | 与现有 TensorBoard / W&B / SwanLab 习惯一致 |
| 与现有后端关系 | 并行输出 | 保持已有实验工作流不变 |
| 第一阶段采集范围 | 文本日志 + scalar + timings | 足够支撑观察性汇总，且改动可控 |
| 分布式策略 | 默认仅 rank 0 上报 | 与现有文件日志策略一致，避免重复 |
| 敏感字段处理 | 不原样上传完整 `opt` | 降低 token 与其他敏感配置泄漏风险 |

## 12. 范围外

以下内容明确不属于本次设计范围：
- 用 Logfire 替代 TensorBoard / W&B / SwanLab
- 将图像样本上传到 Logfire
- 接管 `KAIR/`、`spike_stream_reconstruct/` 等独立 logging 入口
- 调整训练循环的日志调用接口
- 引入额外的 tracing/span 体系
