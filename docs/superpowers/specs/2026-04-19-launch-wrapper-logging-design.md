# Launch Wrapper Logging Design

## Goal

让 [launch_train.sh](/home/wuhy/projects/S-VRT/launch_train.sh) 成为训练入口的统一外层日志兜底层，在不改动现有训练内 Logfire 集成逻辑的前提下，将 shell 输出、`torchrun` 父进程输出、平台注入环境下的启动输出，以及未被训练代码主 logger 捕获的 traceback，一并纳入项目现有主 logger 管理体系。

## Scope

本次设计只覆盖 `launch_train.sh` 启动链路的日志采集与转发，不修改训练代码中已有的 Logfire bridge、`Logger` 类、训练指标上报逻辑，也不调整训练主循环中的已有 `logger.info(...)` 调用方式。

## Current Logging Model

项目当前原生日志入口集中在 [utils/utils_logger.py](/home/wuhy/projects/S-VRT/utils/utils_logger.py:83) 的 `logger_info()`：

- 创建单一 `logging.Logger`
- 同时挂载文件 handler 与终端 handler
- 格式固定为 `'%(asctime)s.%(msecs)03d : %(message)s'`
- 当提供 `opt` 时，为该 logger 挂载 Logfire handler

训练代码在 [main_train_vrt.py](/home/wuhy/projects/S-VRT/main_train_vrt.py:176) 中初始化 `train` logger，之后所有训练核心日志都走这一 logger。因此，现有系统已经有清晰的“主 logger”概念，本次不应再新增平行 logger 体系。

## Problem Statement

当前无法进入 Logfire 的日志主要来自训练入口外层：

- `launch_train.sh` 中的 `echo`
- `torchrun` 启动器输出
- 平台注入环境下 shell 层输出
- 子进程 `stderr`
- Python 未捕获异常 traceback（若未通过 `logger.exception(...)` 进入训练主 logger）

这些信息目前直接停留在 shell stdout/stderr 中，没有进入主 logger，因此也不会经过现有 Logfire handler。

## Design Decision

### 1. 不新增独立 logger 路径

wrapper 日志不创建新的项目级 logger，不建立第二套 Logfire 实现，不引入并行日志基础设施。

所有 launch 层兜底日志最终都写入现有主 logger，由主 logger 统一负责：

- 文件落盘
- 终端显示
- 现有 Logfire handler 转发

### 2. 让 launch wrapper 作为主 logger 的扩展入口存在

在 `launch_train.sh` 中新增一个统一的 wrapper 采集层，但其职责仅为：

- 执行目标命令
- 同时采集 stdout 与 stderr
- 保留终端实时输出
- 将采集到的每一行文本转为主 logger 的一条日志事件

wrapper 自身不负责决定最终写到哪里，也不直接面向 Logfire。它只是主 logger 的上游文本入口。

### 3. 用结构化上下文区分来源，而不是新增 logger 名字

日志来源区分不靠新 logger 实例，而靠写入主 logger 时附加的结构化字段。

建议附加字段：

- `log_origin`: `launch_wrapper` 或 `train_core`
- `launch_stream`: `stdout` 或 `stderr`
- `launch_phase`: `prepare` 或 `train`
- `launch_mode`: `platform_ddp`、`local_single`、`local_multi`
- `launch_command`: 实际执行命令

这样既不污染现有日志文本格式，也能让 Logfire 侧按字段筛选。

## Architecture

### Shell Layer

在 [launch_train.sh](/home/wuhy/projects/S-VRT/launch_train.sh) 中新增统一执行函数，例如 `run_with_wrapper`。

其输入包括：

- 执行阶段：`prepare` / `train`
- 启动模式：`platform_ddp` / `local_single` / `local_multi`
- 命令数组

其行为包括：

- 创建单次运行的 wrapper 日志文件
- 启动目标命令
- 分别采集 stdout / stderr
- 将原始输出保持实时打印到终端
- 对每一行调用“写入主 logger”的桥接入口
- 在命令结束后返回原始退出码

### Python Bridge Layer

新增一个很薄的桥接入口，用于把 shell 传入的文本写入项目主 logger。

这个桥接入口不重新实现项目日志系统，只做以下事情：

- 初始化或获取现有主 logger
- 使用现有 formatter / handler 体系
- 按 stream 选择日志级别：
  - stdout -> `logger.info(...)`
  - stderr -> `logger.error(...)`
- 将结构化上下文通过 `extra` 传入

该桥接入口可放入现有 [utils/utils_logger.py](/home/wuhy/projects/S-VRT/utils/utils_logger.py) 中，作为主 logger 的辅助方法，而不是新建平行 logger 模块。

## Why A Small Python Bridge Still Exists

虽然本设计反对新增并行 logger 路径，但仍然需要一个很小的 Python 桥接入口。原因不是为了新建日志系统，而是因为：

- shell 本身无法直接复用项目现有 `logging.Logger` 与其 handlers
- shell 也无法直接复用现有 Logfire handler
- 若希望 launch 层日志进入当前主 logger，就必须有一个入口把 shell 文本变成 Python `logging` 事件

因此，这里的 Python 代码是“接入现有主 logger 的桥”，不是“第二套日志实现”。

## Logging Semantics

### 训练内核心日志

保持不变：

- 继续使用现有 `logger.info(...)`
- 继续由训练代码决定 message 内容
- 继续按已有格式写文件与终端
- 继续由现有 Logfire handler 转发

### launch wrapper 兜底日志

新增语义：

- shell 层 stdout 作为普通运行信息
- shell 层 stderr 作为错误级别文本
- 若 shell 输出本身已经包含 traceback，则原样逐行写入主 logger
- 不要求重写 traceback 内容，只要求其进入主 logger

## Output Format Strategy

保持现有 formatter 文本格式不变：

`'%(asctime)s.%(msecs)03d : %(message)s'`

原因：

- 避免影响已有训练日志文件格式
- 避免影响已有使用者的日志阅读习惯
- 将来源区分主要放在结构化字段，而非强行修改主 message

若有必要，可只对 wrapper 日志加轻量前缀，例如：

- `[launch][stdout] ...`
- `[launch][stderr] ...`

但这不是首选。首选仍是依赖 `extra` 字段与 Logfire 结构化字段来区分。

## Failure Handling

wrapper 不能吞错误，也不能改变既有启动语义：

- 目标命令退出码必须原样返回
- [launch_train.sh](/home/wuhy/projects/S-VRT/launch_train.sh) 中现有 `handle_error()` 逻辑继续负责交互式停留和最终提示
- wrapper 仅增加采集，不改变流程控制

## Coverage

统一通过 wrapper 覆盖以下路径：

- 数据准备命令
- 平台注入环境下的 `python -u main_train_vrt.py`
- 本地单卡 `python main_train_vrt.py`
- 本地多卡 `torch.distributed.run`

## Non-Goals

本次不包含：

- 修改训练主循环中的日志写法
- 把训练内所有 `print(...)` 全部改写为 `logger.*`
- 调整现有 Logfire 指标上报协议
- 改造 `main_train_vrt.py` 的异常处理逻辑
- 为 shell 输出设计新的日志文件体系

## Acceptance Criteria

满足以下条件即视为完成：

1. 通过 `launch_train.sh` 启动训练时，shell stdout/stderr 都能进入主 logger。
2. `torchrun` 父进程输出和平台注入环境下启动输出都经过同一套 wrapper。
3. 不新增独立项目级 logger 路径。
4. 训练内现有 Logfire 逻辑无需改动即可继续工作。
5. Logfire 中能通过结构化字段区分 `train_core` 与 `launch_wrapper` 来源。
6. `launch_train.sh` 的既有退出码与 `handle_error()` 行为保持不变。
