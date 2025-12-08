# 如何正确运行测试脚本

## 问题：终端闪退

如果运行 `source launch_test.sh` 后终端闪退，这是因为：
1. **不应该使用 `source`**：`source` 用于加载环境变量，不是执行脚本
2. 脚本需要**执行权限**才能直接运行

## 正确的运行方式

### 方法 1：直接执行（推荐）

```bash
# 激活 conda 环境
conda activate vrtspike

# 直接执行脚本（不要用 source）
./launch_test.sh 1 options/gopro_rgbspike_local.json
```

### 方法 2：使用 bash 执行

```bash
conda activate vrtspike
bash launch_test.sh 1 options/gopro_rgbspike_local.json
```

### 方法 3：如果脚本没有执行权限

```bash
# 先添加执行权限
chmod +x launch_test.sh

# 然后执行
./launch_test.sh 1 options/gopro_rgbspike_local.json
```

## 一行命令（推荐）

```bash
conda activate vrtspike && ./launch_test.sh 1 options/gopro_rgbspike_local.json
```

## 如果仍然闪退

1. **检查脚本权限**：
   ```bash
   ls -l launch_test.sh
   chmod +x launch_test.sh  # 如果没有执行权限，添加权限
   ```

2. **检查错误信息**：
   在脚本开头添加调试信息，或使用：
   ```bash
   bash -x launch_test.sh 1 options/gopro_rgbspike_local.json
   ```

3. **检查 conda 环境**：
   ```bash
   conda activate vrtspike
   which python
   python --version
   ```

## 常见错误

- ❌ `source launch_test.sh` - 错误，会闪退
- ✅ `./launch_test.sh` - 正确
- ✅ `bash launch_test.sh` - 正确
