# 数据加速功能状态总结

**更新时间**: 2024-09-09  
**相关文档**: `HANDOFF_DATA_LOADING_ACCELERATION.md`, `EXECUTION_PLAN.md`

---

## 已完成的加速功能（✅ 已在代码中）

### 1. VideoDataset 优化
**文件**: `core/dataset/dataloader_video.py`

- ✅ 输入索引、标签、JPEG 文件列表和 feature 缓存
- ✅ memmap 句柄按 dataset/split/layout 复用（减少重复打开）
- ✅ memmap 样本保留连续帧数组（避免 `np.split` 产生逐帧对象）
- ✅ JPEG 样本一次性 `np.stack`（减少复制）
- ✅ `precache_file_lists` / `cache_file_lists` / `preopen_memmap` 支持
- ✅ `__getstate__` 兼容 spawn worker（不序列化 live memmap）

### 2. GPU 数据路径
**文件**: `core/libs/gpu_video_augmentation.py`, `core/manager/cuda_prefetcher.py`

- ✅ Batch crop/flip/resize/normalize（B,T,C,H,W 格式）
- ✅ CUDA stream 预取下一 batch
- ✅ Non-blocking host-to-device copy
- ✅ CUDA 不可用时自动回退到 CPU
- ✅ 时序 rescale 留在 worker（因为改变时序长度）

### 3. DataLoader 配置增强
**文件**: `core/manager/dataloader_manager.py`, `core/manager/collect_manager.py`

- ✅ 训练/评估分开配置 worker（eval 默认 0）
- ✅ `prefetch_factor` / `persistent_workers` / `pin_memory` 支持
- ✅ Worker 内限制 PyTorch/OpenCV 线程数（避免并发争抢）
- ✅ `CollectManager` 预分配 padded batch（减少临时 Tensor）
- ✅ 可选 `LengthBucketBatchSampler`（默认关闭，DDP 不用）

### 4. 配置参数
**已添加到**: `core/configs/unified_*.yaml`

实验级参数：
```yaml
eval_num_worker: 0
prefetch_factor: 2
persistent_workers: true
pin_memory: true
worker_threads: 1
preopen_memmap: true
gpu_prefetch: true
length_bucket_size: 0  # 0=关闭，4/8=启用 bucketing
```

Feeder 级参数：
```yaml
feeder_args:
    cache_file_lists: true      # 缓存文件列表
    cache_features: true         # 缓存 feature
    precache_file_lists: false   # 预缓存（适合大量小文件）
    gpu_augment: true            # GPU 数据增强
```

---

## 待服务器验证的功能（⏳ 未测试）

根据交接文档，以下功能**已实现但未在 GPU 环境验证**：

1. ⏳ 训练或 1-epoch 冒烟实验
2. ⏳ 真实数据 DataLoader 读取（memmap/JPEG 双路径）
3. ⏳ GPU augment 实际运行
4. ⏳ CUDA prefetch benchmark
5. ⏳ 服务器 IO 吞吐和多实验并发压测

**原因**: 本机无 GPU，未执行上述测试。

---

## 服务器验证计划

### 验证顺序（按交接文档建议）

#### 第 1 步：默认配置短实验
```bash
python main.py --config configs/unified_phoenix2014.yaml \
    --num_epoch 1 \
    --work-dir /tmp/baseline_test
```

**观察指标**:
- GPU 利用率（目标 > 80%）
- Host-to-Device 传输瓶颈
- 共享盘吞吐（iostat）
- Worker 内存占用（nvidia-smi）

#### 第 2 步：Worker 配置扫描
```bash
for workers in 0 2 4 8; do
    python main.py --config configs/unified_phoenix2014.yaml \
        --num-worker $workers \
        --num_epoch 1
done
```

**决策**:
- GPU 利用率低 → 增加 worker
- 内存爆满 → 减少 worker
- 共享盘压力高 → 数据落本地 NVMe，控制总 worker 数

#### 第 3 步：数据类型对比
```bash
# Memmap（推荐）
--feeder-args '{"datatype": "memmap"}'

# JPEG（fallback）
--feeder-args '{"datatype": "video"}'
```

**决策**: 优先使用 memmap，除非文件缺失或损坏。

#### 第 4 步：GPU 加速路径
```bash
# 确认 CUDA 环境
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi

# 启用 GPU 加速
python main.py --config configs/unified_phoenix2014.yaml \
    --gpu-augment true \
    --gpu-prefetch true \
    --num_epoch 1
```

**对比**: 与关闭 GPU 加速的基线比较吞吐量提升。

#### 第 5 步：Length Bucketing（可选）
```bash
# 仅当不需要严格复现采样顺序时使用
--length-bucket-size 4  # 或 8
```

**决策**: 如果 padding 浪费严重（变长视频多），启用 bucketing。但会改变采样顺序，影响复现性。

#### 第 6 步：多实验并发压测
```bash
# 同时启动 2-3 个训练
python main.py --config config1.yaml --device 0 &
python main.py --config config2.yaml --device 1 &
```

**观察**: 总 IO 吞吐、GPU 竞争、共享盘压力。

---

## 优化决策树

根据验证结果调整配置：

### 场景 1: 共享盘压力高
**症状**: iostat 显示磁盘 util% 接近 100%，大量等待 IO

**解决方案**:
- 数据拷贝到本地 NVMe
- 控制 worker 总量（所有实验）
- 启用 `cache_file_lists: true`

### 场景 2: GPU 利用率低（< 80%）
**症状**: GPU 经常空闲，等待数据

**解决方案**:
- 增加 `num_worker`（试 4 → 8）
- 启用 `persistent_workers: true`
- 启用 `gpu_prefetch: true`
- 检查是否 CPU 瓶颈（top 查看）

### 场景 3: 内存占用过高
**症状**: OOM 错误，或 swap 被使用

**解决方案**:
- 减少 `num_worker`
- 减少 `prefetch_factor`
- 使用 memmap 而非完全加载到内存
- 检查是否有内存泄漏

### 场景 4: Padding 浪费严重
**症状**: 变长视频多，batch 中大量 padding

**解决方案**:
- 启用 `length_bucket_size: 4` 或 `8`
- **注意**: 会改变采样顺序，不适合严格复现

### 场景 5: 严格复现要求
**要求**: 每次运行结果必须完全相同

**配置**:
- `length_bucket_size: 0`（关闭 bucketing）
- 固定 `num_worker`
- 固定 `random_seed: 0`
- 记录完整配置到实验报告

---

## 最终推荐配置（待验证后确定）

### 共享盘环境（NFS/Lustre）
```yaml
num_worker: 4
eval_num_worker: 0
prefetch_factor: 2
persistent_workers: true
pin_memory: true
worker_threads: 1
preopen_memmap: true
gpu_prefetch: true
length_bucket_size: 0

feeder_args:
    datatype: memmap
    cache_file_lists: true
    precache_file_lists: false
    gpu_augment: true
```

### 本地 NVMe 环境
```yaml
num_worker: 8
eval_num_worker: 0
prefetch_factor: 4
persistent_workers: true
pin_memory: true
worker_threads: 1
preopen_memmap: true
gpu_prefetch: true
length_bucket_size: 0

feeder_args:
    datatype: memmap
    cache_file_lists: true
    precache_file_lists: true  # 可以更激进
    gpu_augment: true
```

### 调试/快速验证
```yaml
num_worker: 2
eval_num_worker: 0
prefetch_factor: 2
persistent_workers: false
pin_memory: true
worker_threads: 1
preopen_memmap: false
gpu_prefetch: false
length_bucket_size: 0

feeder_args:
    datatype: memmap
    cache_file_lists: false
    gpu_augment: false
```

---

## 性能预期

基于交接文档的设计目标：

| 优化项 | 预期提升 |
|--------|---------|
| memmap 预打开 + 连续帧数组 | 10-20% 吞吐 |
| GPU augmentation | 5-15% 吞吐 |
| CUDA prefetching | 5-10% 吞吐 |
| Persistent workers | 减少 epoch 间延迟 |
| Length bucketing | 10-30% GPU 利用率（变长场景）|
| **总计（理想情况）** | **30-50% 吞吐提升** |

**注意**: 实际提升取决于瓶颈位置（IO vs CPU vs GPU）。

---

## 待办事项

### 立即执行（服务器上）
- [ ] 运行第 1-6 步验证计划
- [ ] 记录每个配置的性能指标
- [ ] 确定最优配置并更新模板

### 文档更新
- [ ] 将验证结果写入 `README.md`（推荐配置）
- [ ] 更新 `docs/PROTOCOLS.md`（如果影响复现性）
- [ ] 在 `EXECUTION_PLAN.md` 中标记完成状态

### 论文素材
- [ ] 性能对比表格（优化前 vs 优化后）
- [ ] 吞吐量曲线（不同 worker 配置）
- [ ] GPU 利用率对比

---

## 参考文档

- `HANDOFF_DATA_LOADING_ACCELERATION.md` - 完整交接文档
- `EXECUTION_PLAN.md` - 执行计划（已整合）
- `core/configs/unified_*.yaml` - 配置模板
