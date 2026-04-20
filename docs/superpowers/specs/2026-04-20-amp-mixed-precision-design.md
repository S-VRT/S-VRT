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
- 仅修复一处自定义 LayerNorm 的数值稳定性问题

---

## 第一节：配置文件

在 `options/*.json` 的 `train` 块新增字段：

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

### sea_raft.py 自定义 LayerNorm

`channels_first` 分支是手动实现的归一化，不在 PyTorch autocast 的自动提升范围内，需显式转 fp32：

```python
def forward(self, x):
    if self.data_format == "channels_last":
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    elif self.data_format == "channels_first":
        x = x.float()
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x.to(self.weight.dtype)
```

### 其他模块无需修改

| 模块 | 原因 |
|------|------|
| `nn.LayerNorm`（stages.py） | PyTorch autocast 自动提升到 fp32 |
| attention softmax（attention.py） | PyTorch autocast 自动提升到 fp32 |
| CharbonnierLoss / SSIM（loss.py） | 在 autocast 外计算，天然 fp32 |
| DCNv4 CUDA kernel | 已有 fp16→fp32 转换逻辑 |

---

## 改动文件汇总

| 文件 | 改动内容 |
|------|---------|
| `options/*.json` | 新增 `amp_dtype` 字段 |
| `models/model_plain.py` | AMP 初始化、optimize_parameters、checkpoint |
| `models/model_vrt.py` | optimize_parameters 同步 AMP 逻辑 |
| `models/optical_flow/sea_raft.py` | channels_first LayerNorm fp32 cast |
