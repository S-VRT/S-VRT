# Stage C — LoRA 微调设计

**状态**：已设计，待实施
**前置**：[2026-04-14-early-fusion-temporal-expansion-design.md](2026-04-14-early-fusion-temporal-expansion-design.md) §5.2；相应 plan 的 Stage C 段已标 Deferred。配置字段 `train.use_lora / lora_rank / lora_alpha / lora_target_modules` 已在 [options/gopro_rgbspike_local.json](../../../options/gopro_rgbspike_local.json) 和 [options/gopro_rgbspike_server.json](../../../options/gopro_rgbspike_server.json) 预留。

## 1. 目标

Stage A 已验证 fusion 方向后，引入 LoRA 微调 VRT backbone：
- **冻结 VRT 主干权重**（`freeze_backbone: true`），仅训练注入的 LoRA adapter（低秩 A/B）+ fusion adapter。
- 训练结束自动导出**合并后（merged）**的 ckpt 作为交付物，其 state_dict 与原 VRT 完全同构，可被非 LoRA 代码路径直接加载。

> Spec §5.2 原文写的是「全参数 + LoRA」，实践中意义不大。本设计固定为「冻结 + LoRA」组合。

## 2. 范围

- **注入对象**：VRT attention 模块 ([models/architectures/vrt/attention.py](../../../models/architectures/vrt/attention.py)) 中的 `qkv_self`、`qkv_mut`、`proj` 三个 `nn.Linear`。匹配通过 config `lora_target_modules`（默认 `["qkv", "proj"]`）的子串匹配在 `named_modules()` 上实现。
- **不注入**：MLP、spike encoder、fusion adapter、光流、DCN。
- **参数预算**：rank=8 时，LoRA 参数 ≈ backbone 的 1-3%。

## 3. 架构

### 3.1 新文件 `models/lora.py`

```python
class LoRALinear(nn.Module):
    """包装 nn.Linear: y = base(x) + (alpha/rank) * B(A(x))

    初始化：A 用 Kaiming uniform，B 用 zeros → 初始前向严格等于 base。
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base            # 冻结原层，保留预训练权重
        self.lora_A = nn.Linear(base.in_features,  rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scaling

    def merged_linear(self) -> nn.Linear:
        """返回一个权重已合并的普通 nn.Linear（shape 与 base 相同）。"""
        fused = nn.Linear(self.base.in_features, self.base.out_features,
                          bias=self.base.bias is not None)
        delta = self.lora_B.weight @ self.lora_A.weight * self.scaling
        fused.weight.data = self.base.weight.data + delta
        if self.base.bias is not None:
            fused.bias.data = self.base.bias.data.clone()
        return fused


def inject_lora(model: nn.Module, target_substrings, rank, alpha) -> list[str]:
    """递归遍历 model.named_modules()，对 isinstance(m, nn.Linear) 且
    模块名（最后一段）含任一 target 子串者，用 setattr(parent, attr, LoRALinear(m, ...))
    原位替换。返回被替换模块的点路径列表（用于 merge 时溯源）。
    """


def merge_lora(model: nn.Module) -> nn.Module:
    """对 model 中所有 LoRALinear 节点，用 merged_linear() 原位替换回 nn.Linear。
    返回修改后的 model（就地修改，同时返回便于链式）。
    """
```

### 3.2 集成到 [models/model_plain.py](../../../models/model_plain.py)

**`init_train()` 中（在 `self.load()` 之后，现有 `freeze_backbone` 调用之前）插入注入**：

```python
self.load()
if self.opt.get('train', {}).get('use_lora', False):
    bare_model = self.get_bare_model(self.netG)
    from models.lora import inject_lora
    inject_lora(bare_model,
                self.opt['train'].get('lora_target_modules', ['qkv', 'proj']),
                self.opt['train'].get('lora_rank', 8),
                self.opt['train'].get('lora_alpha', 16))
if self.opt.get('train', {}).get('freeze_backbone', False):
    freeze_backbone(bare_model)
```

**修改 `freeze_backbone()`**（[models/model_plain.py:18](../../../models/model_plain.py#L18)）：
在现有「保留 fusion_adapter / fusion_operator」白名单之外，新增：若参数路径包含 `lora_A` 或 `lora_B`，保持 `requires_grad=True`。

**新增 `save_merged()` 方法**：

```python
def save_merged(self, iter_label):
    """训练末尾导出合并后的 state_dict，与原 VRT 同构。DDP 下仅 rank 0 执行。"""
    if not self.opt.get('train', {}).get('use_lora', False):
        return
    if self.opt.get('rank', 0) != 0:
        return
    import copy
    from models.lora import merge_lora
    for net, tag in [(self.netG, 'G'), (getattr(self, 'netE', None), 'E')]:
        if net is None: continue
        net_copy = copy.deepcopy(self.get_bare_model(net))
        merge_lora(net_copy)
        path = os.path.join(self.save_dir, f'{iter_label}_{tag}_merged.pth')
        torch.save(net_copy.state_dict(), path)
```

### 3.3 训练入口 [main_train_vrt.py](../../../main_train_vrt.py)

在主训练循环结束后（当前 `current_step >= total_iter`）调用 `model.save_merged(current_step)`。

## 4. Ckpt 策略

| Ckpt | 内容 | 用途 |
|---|---|---|
| 常规 `*_G.pth` | 含 `base.weight`（= 冻结的 VRT 权重）+ `lora_A/B.weight` + fusion 参数 | 中途恢复训练 |
| 常规 `*_E.pth` | 同上，EMA 版本 | 同上 |
| `{total_iter}_G_merged.pth` | 与原 VRT state_dict 完全同构，无 `lora_A/B` 键 | **交付**，可被非 LoRA 路径加载 |
| `{total_iter}_E_merged.pth` | 同上，EMA 版本 | 交付 |

Stage A → Stage C 过渡：
- `pretrained_netG` 指向 Stage A ckpt；现有 `load_network_partial` 按 key+shape 匹配加载，`lora_A/B` 在 Stage A ckpt 中不存在，保持新初始化（A Kaiming、B zeros → 初始 forward 等于 Stage A 结果）。fusion 权重自然继承。

## 5. 测试（新文件 `tests/models/test_lora.py`）

1. **数学等价初始化** — 构造 `LoRALinear(Linear(8,8), rank=4, alpha=8)`，随机输入，断言 `|output - base(x)| < 1e-6`。
2. **Freeze 行为** — 注入到 mini 模型 → 调用 `freeze_backbone` → 断言 `base.weight.requires_grad==False` 且 `lora_A/B.weight.requires_grad==True`；fusion 参数也保持可训。
3. **注入命中** — 构造含 `qkv_self`、`qkv_mut`、`proj` 和一些无关 `nn.Linear` 的 mini 模型，`inject_lora(m, ["qkv","proj"])`，断言命中 3 个、未命中其他。
4. **Merge 保持形状** — 注入、随机设置 A/B 权重、`merge_lora` 后，断言 `state_dict()` 键集与原 VRT 完全相同，每个 tensor shape 相同。
5. **Merge 前后 forward 等价** — 同一输入 + 同样 A/B 权重下，merge 前后输出差 `< 1e-5`。

## 6. 配置示例（Stage C）

已预留字段，仅需切换：

```json
"train": {
    "freeze_backbone": true,
    "partial_load": true,
    "use_lora": true,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["qkv", "proj"]
},
"path": {
    "pretrained_netG": "experiments/<stage_a_run>/models/30000_G.pth"
}
```

## 7. 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 实现路径 | 手写 `LoRALinear` | 无外依赖；state_dict 透明，与现有 `load_network_partial` / EMA / DDP 完全兼容 |
| 注入目标 | 仅 attention `qkv_*` + `proj` | 与 spec 默认一致；参数量控制在 1-3% |
| Freeze 范围 | `freeze_backbone=true` + `use_lora=true` | LoRA 经典用法，显存最低；spec §5.2 原文的「全参数+LoRA」语义混淆，舍弃 |
| Ckpt 存储 | 整合到 netG.state_dict | 无需改 save/load 基础设施 |
| 交付物 | 训练尾自动生成 `*_merged.pth` | 自动化；不影响中间 ckpt 结构 |
| Stage C 初始化 | 继承 Stage A ckpt + LoRA 新初始化（B=0 保证 forward 不变） | 无需特殊加载路径 |

## 8. 范围外

- LoRA 到 MLP、光流、DCN 的注入 — 当前不做。
- Adapter（prefix/prompt tuning 等）其他 PEFT 方法。
- LoRA-only ckpt 单独保存路径 — 不需要（完整 ckpt + 训练尾 merged 已覆盖所有用例）。
