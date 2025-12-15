# Spike 数据结构与帧数记录

针对 GoPro_RGBSpike 数据集中 Spike 片段的结构与帧数实测汇总，仅描述数据本身。

## 路径与对应关系
- RGB 清晰帧示例：`/media/mallm/hd4t/modelrepostore/datasets/gopro_medium/GOPRO_Large/test/GOPR0384_11_00/sharp/*.png`
- Spike 脉冲示例：`/media/mallm/hd4t/modelrepostore/datasets/gopro_medium/GOPRO_Large_spike_seq/test/GOPR0384_11_00/spike/*.dat`
- 同名编号一一对应：`000XYZ.png` ↔ `000XYZ.dat`。每个 `.dat` 内含同一时间片的高频脉冲序列（多于 200 个时间步）。

## 帧数实测（T x H x W）
- `GOPR0384_11_00`（测试集，100 个 `.dat`）
  - `000001`–`000099`: `202 x 250 x 400`
  - `000100`: `186 x 250 x 400`（末段略短）
- `GOPR0854_11_00`（测试集，100 个 `.dat`）
  - `000001`–`000099`: `202 x 250 x 400`
  - `000100`: `186 x 250 x 400`

说明：T 为 `.dat` 内部的 Spike 时间步数量，H/W 为空间分辨率。上述两个 clip 的分辨率一致，末尾第 100 个 `.dat` 时间步略短，其余保持 202 帧。