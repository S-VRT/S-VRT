# Fusion Loss 重设计与 GatedFusionOperator 修复

**日期**：2026-04-21  
**状态**：待实施  
**前置**：[2026-04-19-two-phase-lora-training-design.md](2026-04-19-two-phase-lora-training-design.md)

---

## 1. 问题背景

初步训练（Phase 1 仅 6 iter）发现融合输出图像极度损坏：颜色以红色为主，蓝色作为独立斑块出现，绿色基本消失，但空间结构（花朵轮廓）仍可辨识。与此同时，训练即使收敛，复原帧质量也远低于预训练基线，暗区颜色损失尤为严重。

---

## 2. 根因分析

### RC1 — `rgb_proj` kaiming 初始化破坏颜色通道（直接原因）

当前公式：`out = rgb_proj(rgb) + gate * fuse(concat)`

`rgb_proj = Conv2d(3, 3, kernel=1)` 使用 PyTorch 默认 kaiming_uniform 初始化，权重范围约 ±0.82，等价于对 RGB 施加随机 3×3 线性变换（类随机旋转矩阵）。三个颜色通道被任意线性组合，可能符号反转，这是图像呈红/蓝分离状的直接原因，与训练迭代数无关。

### RC2 — `gate` 和 `fuse` 随机初始化叠加噪声

`gate = Sigmoid(随机 conv 输出)` 均值约 0.5，`fuse` 输出纯随机噪声。公式变为：
```
out ≈ 随机色彩旋转(rgb) + 0.5 × 随机噪声
```
两项都是破坏性的，叠加后效果更严重。

### RC3 — VRT 全局残差连接产生 "回归 blur_rgb" 的吸引子

VRT forward 末尾（`vrt.py:597`）：
```python
return x + x_lq_rgb   # x_lq_rgb = 原始 blur_rgb（fusion 前保存）
```
VRT 预训练为预测残差 `GT - blur_rgb`。Phase 1 中 backbone 冻结，梯度方向为：  
`∂loss/∂fusion_out` 指向"让 VRT 在当前 fusion_out 上预测更准"。  
由于冻结 VRT 在 `fusion_out = blur_rgb` 时预测最准（分布内），因此重建 loss 的梯度**强烈推动 fusion 输出接近 blur_rgb**——即使 RC1/RC2 修复后，这个机制也会在 Phase 1 让 fusion 收敛到"完全忽略 spike"的状态。

### RC4 — fusion 输出用于光流估计（使用 scflow 时已缓解）

`vrt.py:530`：光流估计使用 fusion_out 的前 3 通道。若 fusion 输出损坏，光流错误 → DCN 对齐错误 → 梯度污染。

**注**：当光流模块为 scflow 时，scflow 从 spike 事件流独立估计光流，对 fusion RGB 输出质量不敏感，RC4 在当前配置（scflow）下不构成问题。

---

## 3. 设计方案

### 3.1 架构修改：GatedFusionOperator 改为加法残差结构

**旧公式（有问题）：**
```
out = rgb_proj(rgb) + gate(concat) * fuse(concat)
```

**新公式：**
```
out = rgb + gate(concat) * correction(concat)
```

- **去掉 `rgb_proj`**：base path 变为直接恒等映射（无参数），颜色由设计保证不被破坏
- **`fuse` 重命名为 `correction`**：语义更清晰，表示 spike 对 RGB 的加法修正
- **初始化约束**：
  - `correction` 最后一层（Conv1x1）：权重和 bias 全零初始化 → 初始输出为 0
  - `gate` 倒数第二层（Conv1x1，Sigmoid 前）：bias 初始化为 -5 → Sigmoid(-5) ≈ 0.007

**初始时行为**：`out ≈ rgb + 0.007 × 0 = rgb`，完全 passthrough，颜色完全保留。

**语义约束**：spike 只能通过加法修正贡献信息，不能破坏 RGB 的基础颜色信息。这与 spike 的物理属性一致——spike 不携带颜色，只携带时域运动轮廓。

**兼容性**：改变了公式，现有 checkpoint 中的 `rgb_proj` 参数无法直接迁移。但鉴于现有 checkpoint 本身初始化已损坏，这不构成实质损失。

---

### 3.2 Loss 重设计：Phase 感知的双 loss 策略

#### 核心思路

Phase 1 的 VRT 重建 loss 由于 RC3 的存在，梯度方向与目标（让 fusion 利用 spike 去模糊）相反，需要**完全关闭**。取而代之，对 fusion 的中心帧输出施加直接监督。

#### Fusion Auxiliary Loss 计算

fusion 输出为 `[B, N*S, 3, H, W]`（N=6, S=4 → 24帧）。center frame 提取：
```python
fusion_center = fusion_out[:, S//2 :: S, :, :, :]  # [B, N, 3, H, W]
# S//2 = 2，每组 S 帧取第 2 帧（对应原始 RGB 时间戳）
```

#### Phase 1 Loss（`current_step < fix_iter`）

```
L_phase1 = λ_aux * Charbonnier(fusion_center, GT)
         + λ_pass * Charbonnier(fusion_center, blur_rgb)
```

| 参数 | 默认值 | 作用 |
|------|-------|------|
| `phase1_fusion_aux_loss_weight` (λ_aux1) | 1.0 | Phase 1 主信号：驱动 fusion 向 GT 方向学习（去模糊） |
| `fusion_passthrough_loss_weight` (λ_pass) | 0.2 | Phase 1 正则项：防止 fusion 过度去模糊（保护 Phase 2 VRT residual 不塌缩） |
| `G_lossfn_weight` | 0.0 | Phase 1 关闭 VRT 重建 loss（代码中 `is_phase1` 时覆盖为 0） |

等效监督目标 ≈ `0.83 × GT + 0.17 × blur_rgb`（偏 GT 侧，不完全等于 GT）。

这个配比保证了：fusion 学会利用 spike 去模糊，同时 VRT 在 Phase 2 接收的输入仍有残差改善空间，不会因为输入已接近 GT 而导致 VRT residual target 塌缩。

#### Phase 2 Loss（`current_step >= fix_iter`）

```
L_phase2 = 1.0 * Charbonnier(VRT_LoRA(fusion_out) + blur_rgb, GT)
         + 0.2 * Charbonnier(fusion_center, GT)
```

| 参数 | 默认值 | 作用 |
|------|-------|------|
| `G_lossfn_weight` | 1.0 | VRT 重建 loss 恢复（主信号） |
| `phase2_fusion_aux_loss_weight` | 0.2 | Phase 2 保持 fusion 继续朝 GT 方向微调，防止退化 |
| `fusion_passthrough_loss_weight` | 0.0 | Phase 2 关闭 passthrough 正则 |

---

### 3.3 Fusion 输出 Hook 实现

VRT forward 中，在 fusion 之后存储中间结果：

```python
# vrt.py，fusion 调用之后
x = self.fusion_adapter(rgb=rgb, spike=spike)
self._last_fusion_out = x          # [B, N*S, 3, H, W]
self._last_spike_bins = spike_bins  # S（= spike.shape[2]）
```

model_plain.py 中新增辅助 loss 计算方法：

```python
def _compute_fusion_aux_loss(self, is_phase1: bool):
    if is_phase1:
        aux_weight = self.opt_train.get('phase1_fusion_aux_loss_weight', 0.0)
        pass_weight = self.opt_train.get('fusion_passthrough_loss_weight', 0.0)
    else:
        aux_weight = self.opt_train.get('phase2_fusion_aux_loss_weight', 0.0)
        pass_weight = 0.0
    if aux_weight == 0.0 and pass_weight == 0.0:
        return torch.tensor(0.0, device=self.device)
    vrt = self.get_bare_model(self.netG)
    if not hasattr(vrt, '_last_fusion_out'):
        return torch.tensor(0.0, device=self.device)
    fusion_out = vrt._last_fusion_out
    S = vrt._last_spike_bins
    # 取 center frame（对应原始 RGB 时间戳）
    fusion_center = fusion_out[:, S // 2 :: S, :, :, :]  # [B, N, 3, H, W]
    loss = torch.tensor(0.0, device=self.device)
    if aux_weight > 0.0:
        loss = loss + aux_weight * self.G_lossfn(fusion_center, self.H)
    if pass_weight > 0.0:
        # blur_rgb：dual 模式下 self.L 是 concat([rgb, spike], dim=2)
        # rgb 为前 3 通道
        blur_rgb = self.L[:, :, :3, :, :]
        loss = loss + pass_weight * self.G_lossfn(fusion_center, blur_rgb)
    return loss
```

`optimize_parameters` 中整合：

```python
def optimize_parameters(self, current_step):
    ...
    self.netG_forward()

    # Phase 感知 loss 权重
    is_phase1 = self.fix_iter > 0 and current_step < self.fix_iter
    vrt_loss_weight = 0.0 if is_phase1 else self.G_lossfn_weight

    with self.timer.timer('loss_compute'):
        G_loss = vrt_loss_weight * self.G_lossfn(self.E, self.H)
    G_loss = G_loss + self._compute_fusion_aux_loss(is_phase1=is_phase1)
    ...
```

---

### 3.4 训练迭代数设计

#### 前提分析

- GoPro train set：约 22 个场景 × 240 帧 / 6（num_frame）≈ 880 个序列，加数据增强约 2k 有效样本
- batch_size=2 × 3 GPU = 6 samples/iter（DDP 每卡 2）
- 1 epoch ≈ 2k/6 ≈ 333 iter（非常小）
- fusion 参数量：约 10K（两条 3 层 conv，hidden=32）
- LoRA 参数量：约 1-3% 总参数

#### Phase 1 所需 iter

fusion 从全零修正出发，直接对 GT 监督，任务难度：轻量（single-frame deblur with spike）。  
经验参考：类似规模的小网络在像素 loss 下约 500-2000 iter 初步收敛。  
但需要见到足够多样的场景（覆盖不同运动模糊强度）。  

**推荐：`fix_iter = 6000`**  
- 约 18 个 epoch，覆盖完整数据集多轮
- 相比旧设计（Phase 1 = 20000 iter），节省 14000 iter 的 VRT forward 开销
- 相比设计文档的 6000 iter（两次运行方案），保持一致，符合原有估计

#### Phase 2 所需 iter

LoRA + SCFlow + DCN 适配，任务难度：中等（主干有预训练权重，只需分布迁移）。  
原设计文档：24000 iter。  

**推荐：`total_iter = 30000`，Phase 2 = 24000 iter**（不变）

#### 学习率调度

Phase 1 fusion 理论上需要较高 LR 快速从零出发，但单一 optimizer 无法为 Phase 1/2 独立设置 LR。  
**当前保持 `G_optimizer_lr: 2e-4`**，CosineAnnealing 从 2e-4 衰减至 1e-7，30000 iter 一个完整周期。  
Phase 1 前期 LR 较高（~2e-4），有利于 fusion 快速收敛。

可选优化（本次范围外）：为 Phase 1 单独使用 warmup + 较高峰值 LR，Phase 2 再切换到 CosineAnnealing 从 2e-4 开始。这需要自定义调度器，优先级低。

#### 新旧 iter 分配对比

| | 旧设计（单次运行） | 新设计（单次运行） |
|---|---|---|
| Phase 1 | 20000 iter（VRT 重建 loss，效果差） | 6000 iter（fusion aux loss，高效） |
| Phase 2 | 10000 iter（SCFlow/DCN 解冻） | 24000 iter（SCFlow/DCN 解冻，时间充足） |
| 总计 | 30000 iter | 30000 iter |

---

### 3.5 Config 修改（`gopro_rgbspike_server.json`）

在原 config 基础上修改，不新建文件。主要变更字段：

```json
"train": {
  "G_lossfn_weight": 1.0,                   // Phase 2 生效；Phase 1 代码中 is_phase1=true 时覆盖为 0
  "phase1_fusion_aux_loss_weight": 1.0,      // Phase 1 aux loss 权重（对 GT）
  "phase2_fusion_aux_loss_weight": 0.2,      // Phase 2 aux loss 权重（对 GT，较小）
  "fusion_passthrough_loss_weight": 0.2,     // Phase 1 passthrough 正则权重（对 blur_rgb）

  "fix_iter": 6000,                          // 原值 20000 → 6000
  "fix_keys": ["spynet", "pa_deform"],       // 不变
  "fix_lr_mul": 0.1,                         // 不变

  "total_iter": 30000,                       // 不变
  "G_scheduler_periods": 30000,              // 不变

  "checkpoint_test": [6000, 30000],          // 新增 Phase 1 结束时的验证点
  "checkpoint_save": 2000,                   // 不变
}
```

**`checkpoint_test` 列表化支持**：当前 `checkpoint_test` 为单值整数，需要支持 `[6000, 30000]` 格式，在两个时间点各做一次 val。这需要 `main_train_vrt.py` 或 model 的 checkpoint 检查逻辑将该字段解析为列表。

---

## 4. Phase 1 结束时的期望行为

Phase 1（6000 iter）结束时，fusion 输出应满足：
- 颜色正确（RGB passthrough + 小量修正，初始化保证）
- 图像比 blur_rgb 略锐（spike 修正已介入，但程度可控）
- 与 GT 的 PSNR 高于纯 blur_rgb 基线（否则 aux loss 设计有问题）
- gate 权重分布主要集中在低值区间（运动模糊区域 gate 较高，静态区域接近 0）

验证方式：`checkpoint_test: 6000` 会触发 val，保存的图像直接目视检查 fusion 中心帧质量。

---

## 5. 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `models/fusion/operators/gated.py` | 去掉 `rgb_proj`；`fuse` 改为 `correction`；新增零初始化和 gate bias=-5 初始化 |
| `models/architectures/vrt/vrt.py` | fusion 调用后添加 `self._last_fusion_out` 和 `self._last_spike_bins` 属性赋值 |
| `models/model_plain.py` | 新增 `_compute_fusion_aux_loss()` 方法；修改 `optimize_parameters()` 添加 Phase 感知 loss 权重逻辑 |
| `options/gopro_rgbspike_server.json` | `fix_iter: 6000`；新增 `fusion_aux_loss_weight`、`fusion_passthrough_loss_weight`；`checkpoint_test` 数组化 |
| `main_train_vrt.py` 或 checkpoint 检查逻辑 | 支持 `checkpoint_test` 为列表（`[6000, 30000]`） |

---

## 6. 范围外

- Phase 1 中跳过 VRT forward 以节省计算（改为只跑 EarlyFusionAdapter + aux loss）：可以进一步提速约 8×，但需要在 `netG_forward` 中添加 `phase1_fast_mode` 分支，复杂度较高，当前优先级低
- gate 权重可视化（检查 spike 贡献区域分布）：可用于调试，不影响训练
- fusion_passthrough_loss_weight 的动态退火（Phase 1 后期逐渐降低）：可能比固定 0.2 更优，留作消融
- Phase 2 中 fusion_aux_loss_weight 0.2 vs 0 的消融：观察是否必要保留该项

---

## 7. 关键设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 架构改变 | 去 rgb_proj，改为 additive correction | 从设计上保证颜色 passthrough，不依赖初始化技巧 |
| Phase 1 是否保留 VRT forward | 保留（loss weight=0） | 避免 forward 路径分叉带来的代码复杂性；验证时仍需 VRT forward |
| Phase 1 主 loss 目标 | GT（而非 blur_rgb） | 明确驱动 fusion 利用 spike 去模糊，与 Phase 1 设计意图一致 |
| fusion_passthrough_loss_weight | 0.2 | 防止 fusion 过度去模糊导致 Phase 2 VRT residual 塌缩；可调参数 |
| fix_iter | 6000 | 与两次运行方案的 Phase 1 时长保持一致；节省旧设计 14k iter 的低效训练 |
| total_iter | 30000 | 与旧设计相同；Phase 2 从 10k 扩展到 24k，时间充足 |
| 是否新建 config | 修改原 server.json | 旧 config 存在根本性问题，不值得保留；git 历史保存旧版本 |
| checkpoint_test | [6000, 30000] | Phase 1 结束做一次 val，直接验证 fusion 输出质量 |
