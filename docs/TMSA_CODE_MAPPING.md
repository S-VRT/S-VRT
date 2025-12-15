# TMSA模块代码映射说明

本文档将图中TMSA（Temporal Mutual Self Attention）模块的每个部分映射到项目代码中的具体位置。

## 模块位置

**主文件**: `models/network_vrt.py`
- **TMSA类定义**: 第728-878行
- **WindowAttention类定义**: 第588-726行（包含MRA和MSA实现）
- **SGPBlock类定义**: `models/sgp_vrt.py` 第98-130行（当启用SGP时替换MSA）

---

## 图中各部分对应的代码位置

### 1. 初始LayerNorm（第一个绿色LayerNorm块）

**代码位置**: `models/network_vrt.py` 第786行

```python
self.norm1 = norm_layer(dim)
```

**前向传播中的使用**: 第807行
```python
if not (self.use_sgp and not hasattr(self.attn, 'mut_attn')):
    x = self.norm1(x)
```

**注意**: 当使用SGP时（`use_sgp=True` 且 `mut_attn=False`），LayerNorm在SGPBlock内部处理，此处会跳过。

---

### 2. 并行的MRA和MSA注意力机制

#### 2.1 整体结构

**代码位置**: `models/network_vrt.py` 第795-797行（WindowAttention）或第790-791行（SGPBlock）

```python
# 标准情况：使用WindowAttention
self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, 
                            qkv_bias=qkv_bias, qk_scale=qk_scale, mut_attn=mut_attn)

# SGP情况：使用SGPBlock（当use_sgp=True且mut_attn=False时）
self.attn = SGPBlock(dim=dim, norm_layer=norm_layer, drop_path=drop_path,
                    sgp_w=sgp_w, sgp_k=sgp_k, sgp_reduction=sgp_reduction)
```

**前向传播调用**: 第829-834行
```python
if hasattr(self.attn, 'mut_attn'):
    # WindowAttention case
    attn_windows = self.attn(x_windows, mask=attn_mask)
else:
    # SGPBlock case
    attn_windows = self.attn(x_windows)
```

#### 2.2 MSA（Self-Attention，自注意力）

**代码位置**: `models/network_vrt.py` 第635-639行（WindowAttention.forward方法中）

```python
# self attention
B_, N, C = x.shape
qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
x_out = self.attention(q, k, v, mask, (B_, N, C), relative_position_encoding=True)
```

**相关组件**:
- QKV投影: 第614行 `self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)`
- 相对位置编码: 第610-612行 `self.relative_position_bias_table`
- attention方法: 第657-676行

**注意**: 当`use_sgp=True`且`mut_attn=False`时，MSA被SGPBlock替换（见`models/sgp_vrt.py`）。

#### 2.3 MRA（Mutual Attention，互注意力）

**代码位置**: `models/network_vrt.py` 第642-650行（WindowAttention.forward方法中）

```python
# mutual attention
if self.mut_attn:
    qkv = self.qkv_mut(x + self.position_bias.repeat(1, 2, 1)).reshape(...)
    (q1, q2), (k1, k2), (v1, v2) = torch.chunk(qkv[0], 2, dim=2), ...
    x1_aligned = self.attention(q2, k1, v1, mask, (B_, N // 2, C), relative_position_encoding=False)
    x2_aligned = self.attention(q1, k2, v2, mask, (B_, N // 2, C), relative_position_encoding=False)
    x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)
```

**相关组件**:
- QKV投影: 第621行 `self.qkv_mut = nn.Linear(dim, dim * 3, bias=qkv_bias)`
- 正弦位置编码: 第619-620行 `self.position_bias`（通过`get_sine_position_encoding`生成，第698-725行）

#### 2.4 拼接操作（Concatenation，图中的'C'）

**代码位置**: `models/network_vrt.py` 第650行

```python
x_out = torch.cat([torch.cat([x1_aligned, x2_aligned], 1), x_out], 2)
```

这里将互注意力的输出（`x1_aligned`和`x2_aligned`）与自注意力的输出（`x_out`）拼接。

#### 2.5 投影层（Projection）

**代码位置**: `models/network_vrt.py` 第653行

```python
x = self.proj(x_out)
```

**定义**: 
- 当`mut_attn=True`时: 第622行 `self.proj = nn.Linear(2 * dim, dim)`（因为拼接后维度翻倍）
- 当`mut_attn=False`时: 第615行 `self.proj = nn.Linear(dim, dim)`

---

### 3. 第一个MLP和残差连接（第一个橙色MLP块 + 第一个'+'）

#### 3.1 残差连接（第一个'+'）

**代码位置**: `models/network_vrt.py` 第870行

```python
x = x + self.forward_part1(x, mask_matrix)
```

这是第一个残差连接，将注意力模块的输出与原始输入相加。

#### 3.2 DropPath（在残差连接中）

**代码位置**: `models/network_vrt.py` 第850-851行（forward_part1中）

```python
if not (self.use_sgp and not hasattr(self.attn, 'mut_attn')):
    x = self.drop_path(x)
```

**定义**: 第797行
```python
self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
```

**注意**: 当使用SGP时，DropPath在SGPBlock内部处理，此处会跳过。

---

### 4. 第二个LayerNorm（第二个绿色LayerNorm块）

**代码位置**: `models/network_vrt.py` 第798行

```python
self.norm2 = norm_layer(dim)
```

**前向传播中的使用**: 第856行（在forward_part2中）
```python
return self.drop_path(self.mlp(self.norm2(x)))
```

---

### 5. 第二个MLP和残差连接（第二个橙色MLP块 + 第二个'+'）

#### 5.1 MLP（Feed-Forward Network）

**代码位置**: `models/network_vrt.py` 第799行

```python
self.mlp = Mlp_GEGLU(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)
```

**Mlp_GEGLU类定义**: 第559-585行

**前向传播中的使用**: 第856行
```python
def forward_part2(self, x):
    return self.drop_path(self.mlp(self.norm2(x)))
```

**Mlp_GEGLU内部结构**:
- 第574行: `self.fc11 = nn.Linear(in_features, hidden_features)`
- 第575行: `self.fc12 = nn.Linear(in_features, hidden_features)`
- 第576行: `self.act = act_layer()`（通常是GELU）
- 第577行: `self.fc2 = nn.Linear(hidden_features, out_features)`
- 前向传播（第580-583行）: `x = self.act(self.fc11(x)) * self.fc12(x)`（GEGLU激活）

#### 5.2 残差连接（第二个'+'）

**代码位置**: `models/network_vrt.py` 第876行

```python
x = x + self.forward_part2(x)
```

这是第二个残差连接，将MLP的输出与第一个残差连接的输出相加。

---

## SGP替换说明

当`use_sgp=True`且`mut_attn=False`时，MSA（自注意力）被SGPBlock替换：

### SGPBlock结构

**文件**: `models/sgp_vrt.py` 第98-130行

**内部结构**:
1. **LayerNorm**: 第117行 `self.norm = norm_layer(dim)`
2. **SGP层**: 第118行 `self.sgp = SGP(dim, w=sgp_w, k=sgp_k, reduction=sgp_reduction)`
3. **DropPath**: 第119行 `self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()`
4. **残差连接**: 第129行 `x = x + self.drop_path(self.sgp(self.norm(x)))`

### SGP层结构

**文件**: `models/sgp_vrt.py` 第14-95行

**公式**: `f_SGP(x) = φ(x) · FC(x) + ψ(x) · (Conv_w(x) + Conv_{kw}(x)) + x`

**组成部分**:
- **瞬时分支**（Instant-level）: 第34-42行
  - `fc_main`: 全连接层
  - `fc_instant_gate`: SE风格的MLP门控（φ(x)）
- **窗口分支**（Window-level）: 第44-59行
  - `conv_w_main`: 小卷积核（w）
  - `conv_kw_main`: 大卷积核（w*k）
  - `conv_w_gate`: 窗口级门控（ψ(x)）

**前向传播**: 第61-95行

---

## 完整前向传播流程

```
输入 x: (B, D, H, W, C)
    ↓
forward() 第858行
    ↓
forward_part1() 第801行
    ├─ norm1 (第807行) [或SGPBlock内部处理]
    ├─ window_partition (第826行)
    ├─ WindowAttention/SGPBlock (第829-834行)
    │   ├─ MSA (第635-639行) [或被SGPBlock替换]
    │   ├─ MRA (第642-650行，如果mut_attn=True)
    │   ├─ 拼接 (第650行)
    │   └─ 投影 (第653行)
    ├─ window_reverse (第838行)
    └─ drop_path (第850-851行) [或SGPBlock内部处理]
    ↓
第一个残差连接 (第870行): x = x + forward_part1(x, mask_matrix)
    ↓
forward_part2() 第855行
    ├─ norm2 (第856行)
    ├─ mlp (第856行)
    └─ drop_path (第856行)
    ↓
第二个残差连接 (第876行): x = x + forward_part2(x)
    ↓
输出: (B, D, H, W, C)
```

---

## 关键代码行总结

| 图中组件 | 代码位置 | 说明 |
|---------|---------|------|
| 初始LayerNorm | `network_vrt.py:786, 807` | `self.norm1` |
| MSA（自注意力） | `network_vrt.py:635-639` | WindowAttention.forward中 |
| MRA（互注意力） | `network_vrt.py:642-650` | WindowAttention.forward中（mut_attn=True时） |
| 拼接操作 | `network_vrt.py:650` | `torch.cat` |
| 投影层 | `network_vrt.py:653` | `self.proj` |
| 第一个残差连接 | `network_vrt.py:870` | `x = x + forward_part1(...)` |
| 第二个LayerNorm | `network_vrt.py:798, 856` | `self.norm2` |
| MLP | `network_vrt.py:799, 856` | `self.mlp` (Mlp_GEGLU) |
| 第二个残差连接 | `network_vrt.py:876` | `x = x + forward_part2(x)` |
| SGPBlock（替换MSA） | `sgp_vrt.py:98-130` | 当use_sgp=True且mut_attn=False时 |






