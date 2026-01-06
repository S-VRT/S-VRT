<!-- abc94471-eaa1-471b-8dcd-b1451c6ce71f de24aaa4-502f-44f7-9c40-fef0f65e0e33 -->
# Modularize models into pluggable modules

目标：将单文件 `models/network_vrt.py` 完全前向重构为可复用、按功能分层的模块化代码库，使用配置驱动（registry+factory）实例化模型变体，且不保留任何回滚/临时占位实现。

高层目录变更（新增/修改）示例：

- `models/registry.py` — 模型与 block 注册器
- `models/factory.py` — 从 config 构建模型的入口
- `models/__init__.py` — 对外导出 `build_model`
- `models/blocks/` — 基础构建模块（`basicblock.py`, `dcn.py`, `mlp.py`）
- `models/architectures/vrt/` — VRT 架构子包（`vrt.py`, `stages.py`, `attention.py`, `warp.py`, `reconstruct.py`）
- `models/sgp/` — SGP 实现（`sgp_block.py`）
- `models/utils/` — 公共工具（`init.py`, `windows.py`, `flow.py`）
- `tests/models/` — smoke tests（shape/forward 一致性）

迁移原则（严格遵守）

- 直接前向重构，不保留回滚/legacy 代码。
- 禁止临时占位实现或简化替代，所有模块必须为可运行、完整实现。
- 在实现过程中不得提交 Git（你已声明远程存在）。
- 每一步都必须包含 smoke-test 验证（forward 和 shape）。

迁移步骤（建议顺序）

1. 提取工具函数到 `models/utils/` 并编写单元 smoke tests。  
2. 提取构建 blocks 到 `models/blocks/`，保持接口与原实现一致。  
3. 提取 flow/warp/SpyNet 到 `models/architectures/vrt/warp.py` 并测试输出形状。  
4. 在 `models/architectures/vrt/` 中实现 `stages.py`（Stage/RTMSA/TMSAG）与 `attention.py`（WindowAttention/TMSA 内部）。  
5. 实现 `vrt.py` 作为顶层 facade：组装 conv_first、forward_features、reconstruction。  
6. 实现 `models/registry.py` 与 `models/factory.py`，把 `VRT` 注册为可配置构造体。  
7. 编写 `tests/models/test_vrt_smoke.py`：对比旧实现（在迁移前捕获的 golden tensor）或至少检查形状与无异常前向。  
8. 运行全量 smoke-tests 并修正差异，直到所有测试通过。

验证点

- 所有新模块能在独立导入下 forward 一次并返回正确形状。  
- 使用 `build_model(cfg)` 能通过配置实例化 VRT。  
- 性能/数值差异在可接受范围内（如有必要用统一 RNG 和少量数据对比）。

风险与防范

- 风险：重构后行为差异。防范：每步加 smoke-test，尽早发现差异。  
- 风险：接口命名冲突。防范：为公共接口添加明确文档与类型注释。

迁移后产物（交付物）

- 完整模块化 `models/` 目录（如上）
- `build_model(cfg)` 可从配置切换各变体
- `tests/models/` 下的 smoke tests

若你确认此方向（配置驱动 + 功能分层），我将把上述步骤拆成可执行 todos 并开始实施。

### To-dos

- [x] Extract utility functions into models/utils modules
- [x] Move basic blocks into models/blocks package
- [ ] Place flow/warp and SpyNet into architectures/vrt/warp.py
- [ ] Implement attention and position encodings in attention.py
- [ ] Implement Stage/RTMSA/TMSAG in stages.py
- [ ] Create vrt.py facade assembling full VRT model
- [ ] Add models/registry.py and models/factory.py
- [ ] Add smoke tests under tests/models for forward/shape checks
- [ ] Run smoke tests and resolve numerical/shape mismatches