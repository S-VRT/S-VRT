<!-- d76a2589-aee1-4e01-815f-f6856efbded0 7c1ed85d-a096-4b27-9d97-850baa1f87e0 -->
# Integrate DCNv4 as pluggable DCN module

目标：在不破坏现有功能与架构的前提下，把 DCNv4 并入工程，和现有 DCNv2 并存并可通过配置切换，最终删除外部 `DCNv4-main` 目录。

关键要点：

- 保持向后兼容，所有现有调用点（尤其 `models/blocks/dcn.py` 与 `models/architectures/vrt/stages.py`）在默认配置下行为不变。
- 不留下临时或简化实现；实现必须完整并可编译（包括 CUDA 扩展）。
- 将 `DCNv4-main/DCNv4_op/DCNv4` 的源码完整复制到工程内 `models/op/dcnv4/`（或同层目录），并提供构建脚本。

依赖/影响文件（举例）:

- `models/blocks/dcn.py` （当前 DCNv2 实现）
- `models/architectures/vrt/stages.py` （DCN 使用点）
- `options/` 或项目现有配置路径（添加 `dcn/core_op` 开关）
- `DCNv4-main/DCNv4_op/DCNv4/*` （待复制源码）

短代码引用（示例 DCNv4 构造签名）:

```28:36:/home/mallm/henry/S-VRT/DCNv4-main/DCNv4_op/DCNv4/modules/dcnv4.py
class DCNv4(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
```

实施步骤（高层）:

1. 代码探查：确认项目中所有 DCN 使用点与配置入口，并列出调用签名与数据形状（N,H,W,C vs N,L,C 等）。
2. 新建模块目录：`models/op/dcnv4/`，把 `DCNv4` 源码（Python + C/CUDA ext）完整复制到该目录，保留原始 license 与 README，添加 `__init__.py`。
3. 构建方案：在 `models/op/dcnv4/` 提供 `setup.py`/`build.sh`（与项目 CI 集成），并在项目根 `requirements` 或 `setup` 中注明依赖（torch版本）。
4. 适配包装（兼容层）：实现 `models/blocks/dcn_factory.py`：

   - 暴露统一接口 `get_dcn(name, **kwargs)` 返回实现（DCNv2 wrapper 或 DCNv4 wrapper）。
   - 实现 `DCNv4PackFlowGuided`：保留现有 DCNv2 的 flow-guided offset 生成逻辑，但使用 DCNv4 的核心 `DCNv4` operator（确保输入/输出 shape 匹配）。

5. 修改调用点：在 `models/architectures/vrt/stages.py` 等处，用工厂函数替换直接类创建（最小范围改动并提供回退）。
6. 配置开关：在项目配置（`options` 或现有 config system）添加 `dcn/core_op`（values: `dcnv2`|`dcnv4`），并在模型构建中读取该值。
7. 测试与验证：

   - 单元测试：对比在相同输入下 DCNv2 与 DCNv4 的前向输出 shape 与数值稳定性（小样本）。
   - 集成测试：在训练/推理脚本上运行短时 smoke tests（1 batch），确认无回归。
   - 性能测试：测量单步前向时间与显存占用

8. 文档与移除旧目录：在 PR 中包含迁移说明与构建步骤，CI 成功后删除外部 `DCNv4-main` 并更新 `README`。

回退与安全措施：

- 在替换前保留配置默认值为 `dcnv2`。
- 所有更改先开小范围分支与 CI，确认无误再合并。
- 在 PR 中提供完整编译/安装说明，保证任何用户能复现构建。

示意图（模块关系）:

```mermaid
flowchart LR
  Config["config:dcn/core_op"] -->|select| Factory[get_dcn()]
  Factory --> DCNv2Module["DCNv2 (models/blocks/dcn.py)"]
  Factory --> DCNv4Module["DCNv4 (models/op/dcnv4/*)"]
  Stage["VRT Stage"] -->|calls| Factory
  DCNv4Module -->|requires| CUDAExt["dcnv4 CUDA extension build"]
```

交付物：

- `models/op/dcnv4/`（完整代码 + build scripts）
- `models/blocks/dcn_factory.py`（工厂/兼容包装）
- 小范围调用点改造（`stages.py` 仅替换构造逻辑）
- CI 构建步骤更新 & README 说明

实现风险与缓解：

- 编译失败（CUDA/torch ABI 不兼容）→ 在 CI matrix 中加入常见 torch/cuda 组合，提供 fallback 编译说明。
- 行为差异导致精度回归→ 保持配置可切换，回滚到 `dcnv2`。

### To-dos

- [ ] Map DCN usages and configs
- [ ] Copy DCNv4 sources into models/op/dcnv4
- [ ] Add build scripts for DCNv4 CUDA ext
- [ ] Implement dcn factory and wrappers
- [ ] Replace DCN instantiation at callsites
- [ ] Add tests and CI build jobs
- [ ] Remove external DCNv4-main after merge