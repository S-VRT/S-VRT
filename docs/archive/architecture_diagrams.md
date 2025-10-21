# VRT+Spike 架构可视化图集

本文档提供多种视角的架构可视化图，帮助理解模型的数据流和设计理念。

## 图表目录
1. [高层次架构图](#1-高层次架构图)
2. [详细数据流图](#2-详细数据流图)
3. [通道维度转换图](#3-通道维度转换图)
4. [时序处理流程图](#4-时序处理流程图)
5. [多尺度融合示意图](#5-多尺度融合示意图)

---

## 1. 高层次架构图

這个图展示了模型的核心组件和整体设计理念。

```mermaid
graph LR
    subgraph "输入 Inputs"
        A1[RGB 模糊帧<br/>B×T×3×H×W]
        A2[Spike 体素<br/>B×T×K×H×W]
    end
    
    subgraph "并行编码 Parallel Encoding"
        B1[VRT 编码器<br/>+ TMSA<br/>时序建模]
        B2[SpikeEncoder3D<br/>3D 卷积<br/>時空特征提取]
    end
    
    subgraph "时序增强 Temporal Enhancement"
        C1[RGB 特征<br/>Fr₁..₄<br/>4个尺度]
        C2[Spike 特征<br/>Fs₁..₄<br/>4个尺度]
        C3[SpikeTemporalSA<br/>时序自注意力<br/>Fs → Fs']
    end
    
    subgraph "跨模态融合 Cross-Modal Fusion"
        D1[Cross-Attention<br/>Scale 1: Q=Fr₁ K/V=Fs'₁]
        D2[Cross-Attention<br/>Scale 2: Q=Fr₂ K/V=Fs'₂]
        D3[Cross-Attention<br/>Scale 3: Q=Fr₃ K/V=Fs'₃]
        D4[Cross-Attention<br/>Scale 4: Q=Fr₄ K/V=Fs'₄]
    end
    
    subgraph "解码重建 Decoding"
        E1[VRT 解码器<br/>上采样 + 重建]
        E2[清晰输出<br/>B×T×3×H×W]
    end
    
    A1 --> B1 --> C1
    A2 --> B2 --> C2 --> C3
    
    C1 --> D1
    C1 --> D2
    C1 --> D3
    C1 --> D4
    
    C3 --> D1
    C3 --> D2
    C3 --> D3
    C3 --> D4
    
    D1 --> E1
    D2 --> E1
    D3 --> E1
    D4 --> E1
    E1 --> E2
    
    style A1 fill:#bbdefb
    style A2 fill:#ffecb3
    style B1 fill:#c5e1a5
    style B2 fill:#ffe0b2
    style C3 fill:#e1bee7
    style D1 fill:#ffccbc
    style D2 fill:#ffccbc
    style D3 fill:#ffccbc
    style D4 fill:#ffccbc
    style E1 fill:#c5e1a5
    style E2 fill:#a5d6a7
```

**设计理念**：
- 🔄 **并行处理**: RGB 和 Spike 各自进行時域建模
- 🎯 **渐进融合**: 在多个尺度上进行跨模态信息交互
- ⚡ **高效利用**: Spike 的高时间分辨率补充 RGB 的运动模糊区域

---

## 2. 详细数据流图

展示每个阶段的具體维度转换。

```mermaid
graph TD
    subgraph "Input Stage"
        I1["RGB: (B, T, 3, H, W)<br/>📷 模糊视频序列"]
        I2["Spike: (B, T, K, H, W)<br/>⚡ K=32 时间bins"]
    end
    
    subgraph "Spike Preprocessing"
        S1["维度转换 Permute<br/>(B, T, K, H, W)<br/>↓<br/>(B, K, T, H, W)"]
        S2["投影层 in_proj<br/>Conv3d(32→96, k=3)<br/>↓<br/>(B, 96, T, H, W)"]
    end
    
    subgraph "Multi-Scale Spike Encoding"
        direction TB
        SE1["Scale 1: (B, 96, T, H, W)<br/>✓ 2×ResBlock3D<br/>原始分辨率"]
        SE2["↓ Conv3d stride(1,2,2)<br/>Scale 2: (B, 96, T, H/2, W/2)<br/>✓ 2×ResBlock3D"]
        SE3["↓ Conv3d stride(1,2,2)<br/>Scale 3: (B, 96, T, H/4, W/4)<br/>✓ 2×ResBlock3D"]
        SE4["↓ Conv3d stride(1,2,2)<br/>Scale 4: (B, 96, T, H/8, W/8)<br/>✓ 2×ResBlock3D"]
    end
    
    subgraph "Temporal Self-Attention"
        TSA["SpikeTemporalSA<br/>对每个尺度处理时间维度<br/>Fs₁..₄ → Fs'₁..₄<br/>维度保持不变"]
    end
    
    subgraph "VRT Encoding"
        VE1["VRT Stage 1<br/>(B, 96, T, H, W)<br/>Fr₁"]
        VE2["VRT Stage 2<br/>(B, 96, T', H/2, W/2)<br/>Fr₂"]
        VE3["VRT Stage 3<br/>(B, 96, T'', H/4, W/4)<br/>Fr₃"]
        VE4["VRT Stage 4<br/>(B, 96, T''', H/8, W/8)<br/>Fr₄"]
    end
    
    subgraph "Cross-Modal Fusion"
        F1["CrossAttn-1<br/>Q: Fr₁, K/V: Fs'₁<br/>→ Ff₁"]
        F2["CrossAttn-2<br/>Q: Fr₂, K/V: Fs'₂<br/>→ Ff₂"]
        F3["CrossAttn-3<br/>Q: Fr₃, K/V: Fs'₃<br/>→ Ff₃"]
        F4["CrossAttn-4<br/>Q: Fr₄, K/V: Fs'₄<br/>→ Ff₄"]
    end
    
    subgraph "Output Stage"
        OUT["VRT Decoder<br/>Ff₁..₄ → 上采样<br/>↓<br/>(B, T, 3, H, W)<br/>✨ 清晰输出"]
    end
    
    %% Connections
    I1 --> VE1 --> VE2 --> VE3 --> VE4
    I2 --> S1 --> S2 --> SE1 --> SE2 --> SE3 --> SE4
    
    SE1 --> TSA
    SE2 --> TSA
    SE3 --> TSA
    SE4 --> TSA
    
    VE1 --> F1
    TSA --> F1
    VE2 --> F2
    TSA --> F2
    VE3 --> F3
    TSA --> F3
    VE4 --> F4
    TSA --> F4
    
    F1 --> OUT
    F2 --> OUT
    F3 --> OUT
    F4 --> OUT
    
    style I1 fill:#e3f2fd
    style I2 fill:#fff8e1
    style S1 fill:#fce4ec
    style S2 fill:#f3e5f5
    style SE1 fill:#e0f2f1
    style SE2 fill:#e0f2f1
    style SE3 fill:#e0f2f1
    style SE4 fill:#e0f2f1
    style TSA fill:#ede7f6
    style F1 fill:#ffe0b2
    style F2 fill:#ffe0b2
    style F3 fill:#ffe0b2
    style F4 fill:#ffe0b2
    style OUT fill:#c8e6c9
```

**注意事項**：
- VRT 的时间维度會逐渐压缩（T → T' → T'' → T'''）
- Spike 特征保持时间维度（默认 temporal_stride=1）
- 所有尺度的通道数对齐为 96

---

## 3. 通道维度转换图

聚焦于 Spike 数据如何从 K 个 bins 转换为特征通道。

```mermaid
graph LR
    subgraph "Spike 输入"
        A["体素化 Spike<br/>(B, T, K, H, W)<br/>K = 32 时间bins<br/><br/>示例：<br/>B=2, T=5<br/>K=32, H=256, W=256"]
    end
    
    subgraph "维度重排"
        B["Permute<br/>(B, K, T, H, W)<br/><br/>为3D卷积准备<br/>通道維在前"]
    end
    
    subgraph "通道投影层"
        C["Conv3d(32→96)<br/>kernel_size=3<br/>stride=1<br/>padding=1<br/><br/>学习从32个时间bins<br/>提取96維语义特征"]
    end
    
    subgraph "Spike 特征"
        D["(B, 96, T, H, W)<br/><br/>96通道与VRT对齐<br/>可以进行Cross-Attention"]
    end
    
    A --> |"转置"| B
    B --> |"投影"| C
    C --> |"ReLU + ResBlocks"| D
    
    style A fill:#fff9c4
    style B fill:#f0f4c3
    style C fill:#e1bee7
    style D fill:#c5e1a5
```

**关键设计**：

1. **为什么是 32 bins？**
   - 平衡时间分辨率和计算成本
   - 足夠捕捉快速运动和模糊过程
   - 实验验证的最佳值

2. **为什么投影到 96 通道？**
   - 与 VRT 编码器的特征维度对齐
   - 便于后續的 Cross-Attention 融合
   - 足夠的表達能力而不过度参数化

3. **3D 卷积的优势**
   - 联合处理時空信息
   - 学习时间和空间的局部模式
   - 自然适配体素化的 Spike 数据

---

## 4. 时序处理流程图

对比 RGB 和 Spike 的时序建模策略。

```mermaid
sequenceDiagram
    participant RGB as RGB 分支
    participant VRT as VRT TMSA
    participant Spike as Spike 分支
    participant Enc as SpikeEncoder3D
    participant TSA as SpikeTemporalSA
    participant Fuse as Cross-Attention
    
    Note over RGB,Spike: 阶段 1: 初始编码
    RGB->>VRT: (B, T, 3, H, W)
    Spike->>Enc: (B, T, K, H, W)
    
    Note over VRT: 时间维度建模<br/>TMSA (Temporal Mutual Self-Attention)
    Note over Enc: 3D 卷积提取時空特征
    
    VRT->>VRT: Fr₁..₄ 多尺度特征<br/>时间维度逐渐压缩
    Enc->>TSA: Fs₁..₄ 多尺度特征<br/>时间维度保持
    
    Note over TSA: 阶段 2: Spike 时序增强<br/>沿时间维度 Self-Attention
    TSA->>TSA: 学习长期时间依赖
    TSA->>Fuse: Fs'₁..₄ 时序增强特征
    
    Note over VRT,Fuse: 阶段 3: 跨模态融合
    VRT->>Fuse: Fr₁..₄ (Query)
    
    loop 每个尺度
        Fuse->>Fuse: CrossAttn(Q=Fr, K/V=Fs')
    end
    
    Fuse->>RGB: Ff₁..₄ 融合特征
    
    Note over RGB: 阶段 4: 解码重建
    RGB->>RGB: VRT Decoder
    RGB->>RGB: (B, T, 3, H, W) 清晰输出
```

**时序处理策略对比**：

| 特性 | RGB (VRT TMSA) | Spike (SpikeTemporalSA) |
|------|---------------|------------------------|
| **输入时间分辨率** | T 帧（例如 5 帧） | T 帧 × K bins（例如 5×32 = 160 个时间点） |
| **时间建模方式** | Mutual Self-Attention | Self-Attention |
| **时间维度变化** | 逐渐压缩（T→1） | 保持不变 |
| **建模目标** | 帧間运动和对齐 | 高频运动细节 |
| **计算位置** | VRT 编码器内部 | Spike 编码之后 |

**互补性**：
- RGB: 提供语义和结构信息，但时间分辨率有限
- Spike: 提供高时间分辨率的运动线索，弥补运动模糊

---

## 5. 多尺度融合示意图

展示 4 个尺度上的 Cross-Attention 融合机制。

```mermaid
graph TB
    subgraph "Scale 1 - 原始分辨率 (H×W)"
        S1_RGB["Fr₁<br/>(B, 96, T, H, W)<br/>VRT Stage 1"]
        S1_SPK["Fs'₁<br/>(B, 96, T, H, W)<br/>Spike + TSA"]
        S1_FUSE["Cross-Attention<br/>Q=Fr₁, K/V=Fs'₁<br/>↓<br/>Ff₁: (B, 96, T, H, W)"]
        S1_RGB --> S1_FUSE
        S1_SPK --> S1_FUSE
    end
    
    subgraph "Scale 2 - 1/2 分辨率 (H/2×W/2)"
        S2_RGB["Fr₂<br/>(B, 96, T', H/2, W/2)<br/>VRT Stage 2"]
        S2_SPK["Fs'₂<br/>(B, 96, T, H/2, W/2)<br/>Spike + TSA"]
        S2_FUSE["Cross-Attention<br/>Q=Fr₂, K/V=Fs'₂<br/>↓<br/>Ff₂: (B, 96, T', H/2, W/2)"]
        S2_RGB --> S2_FUSE
        S2_SPK --> S2_FUSE
    end
    
    subgraph "Scale 3 - 1/4 分辨率 (H/4×W/4)"
        S3_RGB["Fr₃<br/>(B, 96, T'', H/4, W/4)<br/>VRT Stage 3"]
        S3_SPK["Fs'₃<br/>(B, 96, T, H/4, W/4)<br/>Spike + TSA"]
        S3_FUSE["Cross-Attention<br/>Q=Fr₃, K/V=Fs'₃<br/>↓<br/>Ff₃: (B, 96, T'', H/4, W/4)"]
        S3_RGB --> S3_FUSE
        S3_SPK --> S3_FUSE
    end
    
    subgraph "Scale 4 - 1/8 分辨率 (H/8×W/8)"
        S4_RGB["Fr₄<br/>(B, 96, T''', H/8, W/8)<br/>VRT Stage 4"]
        S4_SPK["Fs'₄<br/>(B, 96, T, H/8, W/8)<br/>Spike + TSA"]
        S4_FUSE["Cross-Attention<br/>Q=Fr₄, K/V=Fs'₄<br/>↓<br/>Ff₄: (B, 96, T''', H/8, W/8)"]
        S4_RGB --> S4_FUSE
        S4_SPK --> S4_FUSE
    end
    
    DEC["VRT Decoder<br/>Ff₁ + Ff₂ + Ff₃ + Ff₄<br/>↓<br/>上采样 + 重建<br/>↓<br/>(B, T, 3, H, W)"]
    
    S1_FUSE --> DEC
    S2_FUSE --> DEC
    S3_FUSE --> DEC
    S4_FUSE --> DEC
    
    style S1_FUSE fill:#ffe0b2
    style S2_FUSE fill:#ffcc80
    style S3_FUSE fill:#ffb74d
    style S4_FUSE fill:#ffa726
    style DEC fill:#81c784
```

**Cross-Attention 机制细节**：

```
对于每个尺度 i：

1. Query: Fr_i  (来自 VRT RGB 编码)
   - 形状: (B, C, T', H_i, W_i)
   - 含义: "RGB 特征想知道什么信息？"

2. Key/Value: Fs'_i  (来自 Spike + TSA)
   - 形状: (B, C, T, H_i, W_i)
   - 含义: "Spike 能提供什么运动线索？"

3. Attention 计算:
   Q_flat = rearrange(Fr_i, "b c t h w -> b (t h w) c")
   K_flat = rearrange(Fs'_i, "b c t h w -> b (t h w) c")
   V_flat = rearrange(Fs'_i, "b c t h w -> b (t h w) c")
   
   Attention = softmax(Q_flat @ K_flat^T / sqrt(d_k))
   Out_flat = Attention @ V_flat
   
   Ff_i = rearrange(Out_flat, "b (t h w) c -> b c t h w")

4. 输出: Ff_i
   - 形状: (B, C, T', H_i, W_i)
   - 含义: RGB 特征增强了 Spike 的运动信息
```

**为什么多尺度融合？**
- **尺度 1 (高分辨率)**: 捕捉精细的运动细节和边缘
- **尺度 2-3 (中分辨率)**: 平衡细节和全局上下文
- **尺度 4 (低分辨率)**: 提供全局运动模式和场景理解

---

## 附录：3D 卷积可视化

### ResidualBlock3D 结构

```
输入: x ∈ ℝ^(B×C×T×H×W)
  ↓
Conv3d(C→C, kernel=3×3×3, stride=1, padding=1)
  ↓
ReLU
  ↓
Conv3d(C→C, kernel=3×3×3, stride=1, padding=1)
  ↓
  ⊕ ← x (residual connection)
  ↓
ReLU
  ↓
输出: y ∈ ℝ^(B×C×T×H×W)
```

### 3D 卷积的感受野

```
單个 3×3×3 卷积核：
- 时间维度: 3 个时间步
- 空间维度: 3×3 空间区域

经过 N 个残差块后：
- 时间感受野: 2N + 1
- 空间感受野: 2N + 1

例如，2 个 ResBlock3D：
- 时间感受野: 5 个时间步
- 空间感受野: 5×5 像素
```

---

## 总结

### 架构优势

1. **多模态互补** 🔄
   - RGB 提供语义和结构
   - Spike 提供高时间分辨率的运动信息

2. **渐进式融合** 🎯
   - 各模态先獨立完成時域建模
   - 多尺度上进行跨模态信息交互
   - 充分发挥各自优势

3. **灵活性** ⚙️
   - 可调节的通道数和尺度数
   - 支持自适应分块以适应显存限制
   - Monkey-patch 设计便于集成

4. **效率优化** ⚡
   - 3D 卷积高效处理時空数据
   - 梯度检查点减少显存占用
   - Flash Attention 加速（如果可用）

### 关键创新

- ✨ **时序先行**: 兩个分支各自完成时间建模后再融合
- 🎨 **Cross-Attention**: 动态学习跨模态交互，比拼接更灵活
- 🔬 **多尺度**: 在编码器的多个阶段注入 Spike 信息
- 💾 **内存高效**: 自适应分块 + 梯度检查点

---

**参考资料**：
- 主实现: `src/models/integrate_vrt.py`
- 详细文档: `docs/architecture_dataflow.md`
- 配置示例: `configs/deblur/vrt_spike_baseline.yaml`

**生成时间**: 2025-10-20  
**版本**: v2.0





