# AMP 混合精度训练设计

**日期：** 2026-04-20  
**分支：** codex/sync  

---

## 目标

为 S-VRT 项目添加 AMP（Automatic Mixed Precision）混合精度训练支持，通过配置文件开关控制，同时保证数值稳定性。

---

## 方案选择

采用**最小侵入方案**：

- AMP 逻辑集中在 `model_plain.py` 的 `optimize_parameters`
- loss 计算在 autocast 外执行（天然 fp32，无需修改 loss 函数）
- 修复所有不在 autocast 保护范围内的自定义数值敏感操作

---

## 第一节：配置文件

目标文件：
- `options/gopro_rgbspike_server.json`
- `options/006_train_vrt_videodeblurring_gopro_rgbspike.json`

在 `train` 块新增字段：

```json
"train": {
  "amp_dtype": "bf16"
}
```

| 值 | 行为 |
|----|------|
| `"fp16"` | 启用 fp16 AMP，使用 GradScaler |
| `"bf16"` | 启用 bf16 AMP，不使用 GradScaler |
| `null` 或字段缺失 | 禁用 AMP，保持 fp32（向后兼容） |

同时，将两个 config 中的 `G_charbonnier_eps` 从 `1e-9` 改为 `1e-6`（fp16 最小正数约 6e-8，1e-9 下溢为 0）。

---

## 第二节：model_plain.py 集成

### 初始化

在 `__init__` 中根据 `opt_train.get('amp_dtype', None)` 初始化：

```python
amp_dtype_str = opt_train.get('amp_dtype', None)
if amp_dtype_str == 'fp16':
    self.amp_dtype = torch.float16
    self.amp_enabled = True
    self.scaler = torch.cuda.amp.GradScaler()
elif amp_dtype_str == 'bf16':
    self.amp_dtype = torch.bfloat16
    self.amp_enabled = True
    self.scaler = None
else:
    self.amp_dtype = None
    self.amp_enabled = False
    self.scaler = None
```

### optimize_parameters

```python
def optimize_parameters(self, current_step):
    self.G_optimizer.zero_grad()

    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
        self.netG_forward()

    G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

    if self.scaler is not None:
        self.scaler.scale(G_loss).backward()
        self.scaler.step(self.G_optimizer)
        self.scaler.update()
    else:
        G_loss.backward()
        self.G_optimizer.step()
```

### checkpoint 保存/恢复

- `save_training_state`：fp16 时额外保存 `scaler.state_dict()`
- `resume_training`：fp16 时额外恢复 `scaler.load_state_dict()`

---

## 第三节：model_vrt.py

`model_vrt.py` 重写了 `optimize_parameters`，需要同样应用 autocast + scaler 逻辑，结构与 `model_plain.py` 一致。

---

## 第四节：数值稳定性修复

以下操作**不在 PyTorch autocast 自动保护范围内**，需手动修复。

### 4.1 sea_raft.py — channels_first LayerNorm（`models/optical_flow/sea_raft.py`）

`channels_first` 分支是手动实现的归一化，需显式转 fp32 计算：

```python
elif self.data_format == "channels_first":
    x = x.float()
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.eps)
    x = self.weight[:, None, None] * x + self.bias[:, None, None]
    return x.to(self.weight.dtype)
```

同时将 `eps=1e-6` 改为 `eps=1e-5`（在 fp32 计算中仍推荐更保守的 eps）。

### 4.2 sea_raft.py — ConvNextBlock layer_scale 初始化（`models/optical_flow/sea_raft.py:76`）

```python
# 修改前
self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
# 默认 layer_scale_init_value=1e-6

# 修改后：默认改为 1e-4（fp16 最小正数约 6e-5，1e-6 会下溢为 0）
```

将 `ConvNextBlock.__init__` 的默认参数 `layer_scale_init_value=1e-6` 改为 `layer_scale_init_value=1e-4`。

### 4.3 sgp.py — 自定义 LayerNorm（`models/blocks/sgp.py:74-88`）

`SGPBlock` 中的自定义 `LayerNorm` 同样是手动实现，需显式转 fp32：

```python
def forward(self, x):
    assert x.dim() == 3
    assert x.shape[1] == self.num_channels
    x_fp32 = x.float()
    mu = torch.mean(x_fp32, dim=1, keepdim=True)
    res_x = x_fp32 - mu
    sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
    out = res_x / torch.sqrt(sigma + self.eps)
    if self.affine:
        out = out * self.weight.float() + self.bias.float()
    return out.to(x.dtype)
```

### 4.4 base.py — postprocess_flow 位移缩放（`models/optical_flow/base.py:65`）

```python
# 修改前
scale = max_displacement / (displacement + 1e-8)  # 1e-8 在 fp16 下溢为 0

# 修改后
scale = max_displacement / (displacement + 1e-6)
```

### 4.5 loss.py — CharbonnierLoss eps（`models/loss.py:210`）

CharbonnierLoss 在 autocast 外计算（天然 fp32），但 eps 值会从 config 传入，保持代码不变，通过 config 修正：两个 server.json 中的 `G_charbonnier_eps: 1e-9` → `1e-6`。

---

## 第五节：无需修改的模块（已排除）

| 模块 | 原因 |
|------|------|
| `nn.LayerNorm`（stages.py） | PyTorch autocast 自动提升 fp32 |
| `nn.GroupNorm`（sgp.py） | PyTorch autocast 自动提升 fp32 |
| attention softmax（attention.py） | PyTorch autocast 自动提升 fp32 |
| `F.softmax`（pase.py:139） | PyTorch autocast 自动提升 fp32 |
| CharbonnierLoss / SSIM（loss.py） | autocast 外计算，天然 fp32 |
| DCNv4 CUDA kernel | 已有 fp16→fp32 转换逻辑 |
| attention.py 位置编码累加 | 已有 `dtype=torch.float32` 显式保护 |
| corr.py 相关性除法（除以 32） | 结果 0.03125，fp16 安全 |
| snn.py TFP 归一化 | 在 CPU dataloader 中用 numpy/float32，不在 GPU forward 路径 |
| SCFlow wrapper | `torch.no_grad()` + `eval()` 模式，与训练 autocast 隔离 |
| gated.py sigmoid | PyTorch autocast 处理 |
| AffineDropPath init_scale=1e-4 | fp16 可表示（> fp16 最小正数 ~6e-5） |

---

## 改动文件汇总

| 文件 | 改动内容 |
|------|---------|
| `options/gopro_rgbspike_server.json` | 新增 `amp_dtype`，`G_charbonnier_eps` 改为 1e-6 |
| `options/006_train_vrt_videodeblurring_gopro_rgbspike.json` | 同上 |
| `models/model_plain.py` | AMP 初始化、optimize_parameters、checkpoint |
| `models/model_vrt.py` | optimize_parameters 同步 AMP 逻辑 |
| `models/optical_flow/sea_raft.py` | channels_first LayerNorm fp32 cast；layer_scale_init 1e-6→1e-4；eps 1e-6→1e-5 |
| `models/blocks/sgp.py` | 自定义 LayerNorm forward fp32 cast |
| `models/optical_flow/base.py` | postprocess_flow eps 1e-8→1e-6 |
