# OpenCSLR 数据加载加速交接文档

## 目标仓库

- 路径：`/Users/gzhlaker/Documents/code/OpenCSLR`
- 分支：`main`
- 远端：`origin`
- 本次修改只应用于该目录，不涉及 `/Users/gzhlaker/OpenCSLR`。

## 本次改动

### 数据集读取

- `VideoDataset` 增加输入索引、标签、JPEG 文件列表和 feature 缓存。
- memmap 句柄和元数据按数据集、split、布局复用，减少重复打开文件。
- memmap 样本直接保留连续帧数组，避免 `np.split` 产生逐帧 Python 对象。
- JPEG 样本读取后一次性 `np.stack`，减少后续复制。
- 支持 `precache_file_lists` 和 `cache_file_lists`，适合服务器上大量小文件场景。
- 支持 `preopen_memmap`，训练 DataLoader 创建前预打开训练 memmap。
- `__getstate__` 不序列化 live memmap，兼容 spawn worker。

### GPU 数据路径

- `core/libs/gpu_video_augmentation.py`
  - 支持 batch crop、horizontal flip、resize 和归一化。
  - 输入格式为 `B,T,C,H,W`。
- `core/manager/cuda_prefetcher.py`
  - 使用 CUDA stream 预取下一 batch。
  - 使用 non-blocking host-to-device copy。
  - CUDA 不可用时自动退回普通 DataLoader 迭代。
- 时序 rescale 保留在 worker 中，因为它会改变时序长度；空间 augment 在 GPU batch 阶段执行。

### DataLoader 和 batch 整理

- 训练集与评估集可分别配置 worker 数量；评估集默认可使用 0 worker。
- 支持 `prefetch_factor`、`persistent_workers`、`pin_memory`。
- worker 内限制 PyTorch/OpenCV 线程数，避免多实验并发时线程过量争抢 IO/CPU。
- `CollectManager` 改为预分配 padded batch，再通过 slice/copy 写入，减少临时 Tensor 和重复拼接。
- 新增可选 `LengthBucketBatchSampler`，默认关闭；非 DDP 训练时可减少变长视频 padding 浪费。DDP 继续使用原有 `DistributedSampler`。

### 配置

沿用目标仓库现有的三文件配置结构：

- `core/configs/exp.yaml`
- `core/configs/network.yaml`
- `core/configs/dataset.yaml`

新增实验级参数：

```yaml
eval_num_worker: 0
prefetch_factor: 2
persistent_workers: true
pin_memory: true
worker_threads: 1
preopen_memmap: true
gpu_prefetch: true
length_bucket_size: 0
```

新增 feeder 参数：

```yaml
cache_file_lists: true
cache_features: true
precache_file_lists: false
gpu_augment: true
```

`ConfigManager` 已增加嵌套配置非法键和基础类型校验。若服务器环境没有 CUDA，`gpu_augment` 会自动回退为 CPU 路径。

## 服务器侧建议

1. 先用默认配置运行一个短实验，观察 GPU 利用率、host-to-device 利用率、共享盘吞吐和 worker 内存。
2. 共享盘压力仍高时，优先把数据放到本地 NVMe；多实验不要让每个进程无限增加 worker。
3. memmap 数据优先保持 `datatype: memmap`；原始 JPEG 数据量较大时，再评估 NVJPEG/nvImageCodec、DALI 或 WebDataset shard。
4. `length_bucket_size` 可从 `4` 或 `8` 试起；需要严格复现实验采样顺序时保持 `0`。
5. `gpu_augment` 与 `gpu_prefetch` 需要 CUDA 环境；服务器上确认 PyTorch CUDA 版本和显卡驱动匹配后再启用。

## 尚未在本机执行的事项

本机无显卡，且本次按要求停止检查，因此未执行：

- 训练或 1 epoch 冒烟实验；
- 真实数据 DataLoader 读取；
- GPU augment 运行；
- CUDA prefetch benchmark；
- 服务器 IO 吞吐和多实验并发压测。

这些项目应在有 CUDA 和真实数据路径的服务器上完成。
