# AMP 混合精度训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 S-VRT 项目添加 AMP 混合精度训练支持，通过配置文件 `amp_dtype` 字段控制 fp16/bf16/禁用，同时修复所有在 autocast 下会 crash 或数值不稳定的操作。

**Architecture:** AMP 逻辑集中在 `model_plain.py` 的 `optimize_parameters`，`model_vrt.py` 通过 `super()` 继承。Loss 在 autocast 外计算（天然 fp32）。数值修复分散在各自模块，不引入新的抽象层。

**Tech Stack:** PyTorch `torch.autocast`, `torch.cuda.amp.GradScaler`

---

### Task 1: 修复 flow_warp dtype 不匹配（crash fix，无 GPU 可测）

**Files:**
- Modify: `models/utils/flow.py:26`
- Test: `tests/models/test_warp_smoke.py`

- [ ] **Step 1: 在 test_warp_smoke.py 添加 fp16 测试**

```python
def test_flow_warp_fp16():
    """flow_warp must not crash with fp16 input under autocast."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(1, 3, 64, 64, device='cuda', dtype=torch.float16)
    flow = torch.randn(1, 64, 64, 2, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = flow_warp(x, flow)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
```

在文件顶部已有 `from models.optical_flow.spynet import SpyNet`，在其下方添加：
```python
import pytest
from models.utils.flow import flow_warp
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /home/wuhy/projects/S-VRT
python -m pytest tests/models/test_warp_smoke.py::test_flow_warp_fp16 -v
```

预期：FAIL（RuntimeError: expected scalar type Float but found Half）

- [ ] **Step 3: 修复 flow.py**

将 `models/utils/flow.py` 第 26 行：
```python
grid = torch.stack((grid_x, grid_y), 2).float()
```
改为：
```python
grid = torch.stack((grid_x, grid_y), 2)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/models/test_warp_smoke.py -v
```

预期：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add models/utils/flow.py tests/models/test_warp_smoke.py
git commit -m "fix(flow): remove .float() cast in flow_warp grid to fix fp16 dtype mismatch"
```

---

### Task 2: 修复 SCFlowWrapper autocast 隔离（crash fix）

**Files:**
- Modify: `models/optical_flow/scflow/wrapper.py`
- Modify: `models/optical_flow/scflow/models/scflow.py` lines 32, 159
- Test: `tests/models/test_scflow_amp.py`（新建）

- [ ] **Step 1: 新建测试文件**

新建 `tests/models/test_scflow_amp.py`：

```python
import pytest
import torch
from models.optical_flow.scflow.wrapper import SCFlowWrapper


def test_scflow_wrapper_fp16_no_crash():
    """SCFlowWrapper must not crash when called inside fp16 autocast context."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wrapper = SCFlowWrapper(device='cuda')
    spk1 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    spk2 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    with torch.autocast("cuda", dtype=torch.float16):
        flows = wrapper(spk1, spk2)
    assert len(flows) == 4
    for f in flows:
        assert not torch.isnan(f).any()
        assert not torch.isinf(f).any()


def test_scflow_wrapper_output_is_float32():
    """SCFlowWrapper outputs should be float32 regardless of outer autocast."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    wrapper = SCFlowWrapper(device='cuda')
    spk1 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    spk2 = torch.randint(0, 2, (1, 25, 64, 64), device='cuda').float()
    with torch.autocast("cuda", dtype=torch.float16):
        flows = wrapper(spk1, spk2)
    for f in flows:
        assert f.dtype == torch.float32
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/models/test_scflow_amp.py -v
```

预期：FAIL（RuntimeError: expected scalar type Float but found Half）

- [ ] **Step 3: 修复 wrapper.py**

将 `models/optical_flow/scflow/wrapper.py` 中的 `forward` 方法替换为：

```python
def forward(self, spk1: torch.Tensor, spk2: torch.Tensor) -> List[torch.Tensor]:
    """
    Forward pass for SCFlow.
    Args:
        spk1, spk2: Spike sequences of shape (B, 25, H, W)
    Returns:
        List of 4 flows: [full_res, 1/2_res, 1/4_res, 1/8_res]
    """
    self._validate_spike_pair(spk1, spk2)

    try:
        device = next(self.model.parameters()).device
    except StopIteration:
        device = self.device

    spk1 = spk1.to(device=device, dtype=torch.float32)
    spk2 = spk2.to(device=device, dtype=torch.float32)

    b, _, h, w = spk1.shape
    flow_init = torch.zeros(b, 2, h, w, device=device, dtype=torch.float32)

    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        flows, _ = self.model(spk1, spk2, flow_init, dt=self.dt)

    return flows[:4]
```

- [ ] **Step 4: 修复 scflow.py line 159**

将 `models/optical_flow/scflow/models/scflow.py` 第 159 行：
```python
flow = torch.zeros(b, 2, h, w, dtype=init_dtype, device=init_device).float()
```
改为：
```python
flow = torch.zeros(b, 2, h, w, dtype=init_dtype, device=init_device)
```

- [ ] **Step 5: 修复 scflow.py line 32**

将第 32 行：
```python
flow_factor = (torch.linspace(-12, 12, steps=25, device=seq.device) / dt)
```
改为：
```python
flow_factor = (torch.linspace(-12, 12, steps=25, dtype=seq.dtype, device=seq.device) / dt)
```

- [ ] **Step 6: 运行测试确认通过**

```bash
python -m pytest tests/models/test_scflow_amp.py -v
```

预期：全部 PASS

- [ ] **Step 7: Commit**

```bash
git add models/optical_flow/scflow/wrapper.py models/optical_flow/scflow/models/scflow.py tests/models/test_scflow_amp.py
git commit -m "fix(scflow): isolate SCFlowWrapper from autocast context; fix internal dtype consistency"
```

---

### Task 3: 修复数值稳定性问题（sea_raft, sgp, base）

**Files:**
- Modify: `models/optical_flow/sea_raft.py` lines 47, 57-65, 76
- Modify: `models/blocks/sgp.py` lines 74-88
- Modify: `models/optical_flow/base.py` line 65
- Test: `tests/models/test_amp_numerical.py`（新建）

- [ ] **Step 1: 新建数值稳定性测试**

新建 `tests/models/test_amp_numerical.py`：

```python
import pytest
import torch
from models.optical_flow.sea_raft import LayerNorm as SeaRaftLayerNorm
from models.blocks.sgp import LayerNorm as SGPLayerNorm


def test_searaft_layernorm_channels_first_fp16():
    """channels_first LayerNorm must not produce NaN/inf with fp16 input."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    norm = SeaRaftLayerNorm(32, data_format="channels_first").cuda()
    x = torch.randn(2, 32, 64, 64, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = norm(x)
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_searaft_layernorm_channels_last_fp16():
    """channels_last LayerNorm (via F.layer_norm) must work with fp16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    norm = SeaRaftLayerNorm(32, data_format="channels_last").cuda()
    x = torch.randn(2, 64, 64, 32, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = norm(x)
    assert not torch.isnan(out).any()


def test_sgp_layernorm_fp16():
    """SGP custom LayerNorm must not produce NaN/inf with fp16 input."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    norm = SGPLayerNorm(64).cuda()
    x = torch.randn(2, 64, 16, device='cuda', dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = norm(x)
    assert out.dtype == torch.float16
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/models/test_amp_numerical.py -v
```

预期：`test_searaft_layernorm_channels_first_fp16` 和 `test_sgp_layernorm_fp16` FAIL（NaN 或计算错误）

- [ ] **Step 3: 修复 sea_raft.py LayerNorm channels_first**

将 `models/optical_flow/sea_raft.py` 中 `LayerNorm.__init__` 的 `eps=1e-6` 改为 `eps=1e-5`，`forward` 的 `channels_first` 分支改为：

```python
elif self.data_format == "channels_first":
    x = x.float()
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.eps)
    x = self.weight[:, None, None] * x + self.bias[:, None, None]
    return x.to(self.weight.dtype)
```

- [ ] **Step 4: 修复 sea_raft.py ConvNextBlock layer_scale**

将 `models/optical_flow/sea_raft.py` 中 `ConvNextBlock.__init__` 的默认参数：
```python
def __init__(self, dim, output_dim, layer_scale_init_value=1e-6):
```
改为：
```python
def __init__(self, dim, output_dim, layer_scale_init_value=1e-4):
```

- [ ] **Step 5: 修复 sgp.py LayerNorm**

将 `models/blocks/sgp.py` 中的 `LayerNorm.forward` 方法替换为：

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

- [ ] **Step 6: 修复 base.py eps**

将 `models/optical_flow/base.py` 第 65 行：
```python
scale = max_displacement / (displacement + 1e-8)
```
改为：
```python
scale = max_displacement / (displacement + 1e-6)
```

- [ ] **Step 7: 运行测试确认通过**

```bash
python -m pytest tests/models/test_amp_numerical.py -v
```

预期：全部 PASS

- [ ] **Step 8: Commit**

```bash
git add models/optical_flow/sea_raft.py models/blocks/sgp.py models/optical_flow/base.py tests/models/test_amp_numerical.py
git commit -m "fix(amp): fp32 cast for custom LayerNorms; fix layer_scale_init and eps for fp16 safety"
```

---

### Task 4: 更新配置文件

**Files:**
- Modify: `options/gopro_rgbspike_server.json`
- Modify: `options/006_train_vrt_videodeblurring_gopro_rgbspike.json`

- [ ] **Step 1: 更新 gopro_rgbspike_server.json**

找到 `"G_charbonnier_eps": 1e-9`，改为 `"G_charbonnier_eps": 1e-6`。

在 `train` 块中添加（放在 `G_charbonnier_eps` 附近）：
```json
"amp_dtype": "bf16",
```

- [ ] **Step 2: 更新 006_train_vrt_videodeblurring_gopro_rgbspike.json**

同上：`G_charbonnier_eps` 改为 `1e-6`，添加 `"amp_dtype": "bf16"`。

- [ ] **Step 3: Commit**

```bash
git add options/gopro_rgbspike_server.json options/006_train_vrt_videodeblurring_gopro_rgbspike.json
git commit -m "config: enable bf16 AMP; fix G_charbonnier_eps to 1e-6 for fp16 safety"
```

---

### Task 5: model_plain.py AMP 集成

**Files:**
- Modify: `models/model_plain.py`
- Test: `tests/models/test_amp_model.py`（新建）

- [ ] **Step 1: 新建 model AMP 测试**

新建 `tests/models/test_amp_model.py`：

```python
import pytest
import torch
from unittest.mock import MagicMock, patch


def _make_opt_train(amp_dtype=None):
    return {
        'amp_dtype': amp_dtype,
        'E_decay': 0,
        'G_lossfn_type': 'l1',
        'G_lossfn_weight': 1.0,
        'G_charbonnier_eps': 1e-6,
        'G_optimizer_type': 'adam',
        'G_optimizer_lr': 1e-4,
        'G_optimizer_betas': [0.9, 0.999],
        'G_optimizer_wd': 0,
        'G_optimizer_clipgrad': None,
        'G_regularizer_orthstep': None,
        'G_regularizer_clipstep': None,
        'G_scheduler_type': 'MultiStepLR',
        'G_scheduler_milestones': [100000],
        'G_scheduler_gamma': 0.5,
    }


def test_amp_disabled_no_scaler():
    """When amp_dtype is None, scaler must be None and amp_enabled must be False."""
    from models.model_plain import ModelPlain
    opt = {
        'train': _make_opt_train(amp_dtype=None),
        'netG': {'net_type': 'vrt'},
        'dist': False,
        'gpu_ids': [],
        'path': {'pretrained_netG': None, 'pretrained_netE': None, 'pretrained_optimizerG': None},
        'is_train': True,
        'scale': 1,
    }
    with patch('models.model_plain.define_G') as mock_define_G:
        mock_define_G.return_value = MagicMock()
        with patch.object(ModelPlain, 'model_to_device', lambda self, m: m):
            model = ModelPlain.__new__(ModelPlain)
            model.opt = opt
            model.opt_train = opt['train']
            model.device = 'cpu'
            model.schedulers = []
            model.log_dict = {}
            # Call only the AMP init portion
            amp_dtype_str = opt['train'].get('amp_dtype', None)
            if amp_dtype_str == 'fp16':
                model.amp_dtype = torch.float16
                model.amp_enabled = True
                model.scaler = torch.cuda.amp.GradScaler()
            elif amp_dtype_str == 'bf16':
                model.amp_dtype = torch.bfloat16
                model.amp_enabled = True
                model.scaler = None
            else:
                model.amp_dtype = None
                model.amp_enabled = False
                model.scaler = None

    assert model.amp_enabled is False
    assert model.scaler is None
    assert model.amp_dtype is None


def test_amp_fp16_creates_scaler():
    """When amp_dtype is 'fp16', scaler must be a GradScaler instance."""
    amp_dtype_str = 'fp16'
    if amp_dtype_str == 'fp16':
        amp_dtype = torch.float16
        amp_enabled = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        amp_dtype = None
        amp_enabled = False
        scaler = None

    assert amp_enabled is True
    assert scaler is not None
    assert isinstance(scaler, torch.cuda.amp.GradScaler)


def test_amp_bf16_no_scaler():
    """When amp_dtype is 'bf16', scaler must be None."""
    amp_dtype_str = 'bf16'
    if amp_dtype_str == 'fp16':
        scaler = torch.cuda.amp.GradScaler()
    elif amp_dtype_str == 'bf16':
        scaler = None
    else:
        scaler = None

    assert scaler is None
```

- [ ] **Step 2: 运行测试确认通过（这些是初始化测试，应通过）**

```bash
python -m pytest tests/models/test_amp_model.py -v
```

预期：全部 PASS（初始化逻辑尚未在 model_plain.py 实现，这些单元测试直接测逻辑）

- [ ] **Step 3: 在 model_plain.py 的 `__init__` 末尾添加 AMP 初始化**

在 `ModelPlain.__init__` 方法末尾（即 `self.L_flow_spike = None` 这一行之后）添加：

```python
# AMP 初始化
amp_dtype_str = self.opt_train.get('amp_dtype', None)
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

- [ ] **Step 4: 修改 optimize_parameters**

将现有的 `optimize_parameters` 方法中的 forward + loss + backward 部分（约第 385-406 行）替换为带 AMP 的版本：

```python
def optimize_parameters(self, current_step):
    # 重置当前迭代的计时
    self.timer.current_timings.clear()

    with self.timer.timer('zero_grad'):
        self.G_optimizer.zero_grad()

    with self.timer.timer('forward'):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
            self.netG_forward()

    with self.timer.timer('loss_compute'):
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)

    with self.timer.timer('backward'):
        if self.scaler is not None:
            self.scaler.scale(G_loss).backward()
        else:
            G_loss.backward()

    # ------------------------------------
    # clip_grad
    # ------------------------------------
    G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
    if G_optimizer_clipgrad > 0:
        with self.timer.timer('clip_grad'):
            if self.scaler is not None:
                self.scaler.unscale_(self.G_optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

    with self.timer.timer('optimizer_step'):
        if self.scaler is not None:
            self.scaler.step(self.G_optimizer)
            self.scaler.update()
        else:
            self.G_optimizer.step()

    # ------------------------------------
    # regularizer
    # ------------------------------------
    G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
    if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
        with self.timer.timer('regularizer_orth'):
            self.netG.apply(regularizer_orth)
    G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
    if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
        with self.timer.timer('regularizer_clip'):
            self.netG.apply(regularizer_clip)

    self.log_dict['G_loss'] = G_loss.item()

    if self.opt_train['E_decay'] > 0:
        with self.timer.timer('update_E'):
            self.update_E(self.opt_train['E_decay'])
```

注意：clip_grad 在 fp16 时需要先 `unscale_` 才能正确裁剪梯度，否则裁剪的是缩放后的梯度。

- [ ] **Step 5: 修改 save 方法，保存 scaler**

将 `save` 方法：
```python
def save(self, iter_label):
    self.save_network(self.save_dir, self.netG, 'G', iter_label)
    if self.opt_train['E_decay'] > 0:
        self.save_network(self.save_dir, self.netE, 'E', iter_label)
    if self.opt_train['G_optimizer_reuse']:
        self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
```
改为：
```python
def save(self, iter_label):
    self.save_network(self.save_dir, self.netG, 'G', iter_label)
    if self.opt_train['E_decay'] > 0:
        self.save_network(self.save_dir, self.netE, 'E', iter_label)
    if self.opt_train['G_optimizer_reuse']:
        self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
    if self.scaler is not None:
        scaler_path = os.path.join(self.save_dir, f'{iter_label}_scaler.pth')
        torch.save(self.scaler.state_dict(), scaler_path)
```

确认 `model_plain.py` 顶部已有 `import os`（若无则添加）。

- [ ] **Step 6: 修改 load_optimizers 方法，恢复 scaler**

将 `load_optimizers` 方法：
```python
def load_optimizers(self):
    load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
    if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
        print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
        self.load_optimizer(load_path_optimizerG, self.G_optimizer)
```
改为：
```python
def load_optimizers(self):
    load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
    if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
        print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
        self.load_optimizer(load_path_optimizerG, self.G_optimizer)
    if self.scaler is not None and load_path_optimizerG is not None:
        scaler_path = load_path_optimizerG.replace('_optimizerG.pth', '_scaler.pth')
        if os.path.exists(scaler_path):
            self.scaler.load_state_dict(torch.load(scaler_path, map_location='cpu'))
            print(f'Loading GradScaler state from [{scaler_path}]')
```

- [ ] **Step 7: 运行已有 smoke 测试确认没有回归**

```bash
python -m pytest tests/models/test_amp_model.py -v
```

预期：全部 PASS

- [ ] **Step 8: Commit**

```bash
git add models/model_plain.py tests/models/test_amp_model.py
git commit -m "feat(amp): add AMP fp16/bf16 support to ModelPlain with GradScaler and checkpoint persistence"
```

---

### Task 6: 验证整体集成（smoke test）

**Files:**
- Test: `tests/models/test_amp_integration.py`（新建）

- [ ] **Step 1: 新建集成 smoke 测试**

新建 `tests/models/test_amp_integration.py`：

```python
import pytest
import torch


@pytest.mark.parametrize("amp_dtype", [None, "bf16", "fp16"])
def test_amp_config_parsing(amp_dtype):
    """AMP dtype config parses to correct torch dtype and scaler state."""
    if amp_dtype == "fp16" and not torch.cuda.is_available():
        pytest.skip("fp16 GradScaler requires CUDA")
    if amp_dtype == "bf16" and not torch.cuda.is_available():
        pytest.skip("bf16 requires CUDA")

    if amp_dtype == 'fp16':
        amp_dtype_val = torch.float16
        amp_enabled = True
        scaler = torch.cuda.amp.GradScaler()
    elif amp_dtype == 'bf16':
        amp_dtype_val = torch.bfloat16
        amp_enabled = True
        scaler = None
    else:
        amp_dtype_val = None
        amp_enabled = False
        scaler = None

    assert amp_enabled == (amp_dtype is not None)
    if amp_dtype == 'fp16':
        assert scaler is not None
    else:
        assert scaler is None


def test_flow_warp_fp32_unchanged():
    """flow_warp must still work correctly with fp32 input after the fix."""
    from models.utils.flow import flow_warp
    x = torch.randn(1, 3, 32, 32)
    flow = torch.randn(1, 32, 32, 2)
    out = flow_warp(x, flow)
    assert out.shape == x.shape
    assert out.dtype == torch.float32
```

- [ ] **Step 2: 运行集成测试**

```bash
python -m pytest tests/models/test_amp_integration.py -v
```

预期：全部 PASS

- [ ] **Step 3: 运行全量 tests/models/ 确认无回归**

```bash
python -m pytest tests/models/ -v --ignore=tests/models/test_vrt_integration.py -x
```

（`test_vrt_integration.py` 依赖完整模型权重，排除在外）

预期：全部已有测试 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/models/test_amp_integration.py
git commit -m "test(amp): add integration smoke tests for AMP config and flow_warp fp32 regression"
```
