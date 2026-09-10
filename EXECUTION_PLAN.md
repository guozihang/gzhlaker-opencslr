# OpenCSLR v1.0.0 执行计划

**最后更新**: 2026-09-10  
**当前阶段**: 代码集成完成，准备服务器验证与加速调优

---

## 一、立即执行任务（按优先级）

### 1.1 完成剩余代码集成（2-4 小时）✅ 已完成

**落地位置与实现差异**（下文伪代码为设计参考，以实际代码为准）：

| 文件 | 实现 |
|------|------|
| `core/pipline/single.py` | `seq_eval` 用 `SampleStatistics` 记录成功/跳过（`frames_exceeded`）/失败样本，落盘 `sample_statistics_{dev,test}.json`（dev/test 互不覆盖），跳过率 >5% 时告警；返回值仍是 float WER，`ExperimentManager` 无需改动 |
| `core/manager/dataloader_manager.py` | `_worker_init` 在限制线程数之外调用 `utils.seed_utils.seed_worker`，固定 `random_seed` 时 worker 内增强可复现 |
| `core/manager/evaluation_manager.py` | 新增 `save_evaluation_results()`，由 `seq_eval` 调用写出 `experiment_result_{dev,test}.json`（test 同时使用权威名 `experiment_result.json`）；保存失败只记日志，不影响训练 |

本地验证（无需 GPU / 数据集）：

```bash
cd core
python tests/test_stats_integration.py    # 6 项：worker 播种、prefetcher 回退、评估统计与结果落盘
```

**已知遗留**：`configs/unified_*.yaml` 目前无法被 `--config` 直接加载（`ArgumentManager.map` 不认识 `experiment_name`/`protocol`/`decoder_args`，且 `ConfigManager` 只支持 `--exp <节名>` + `network:` 引用）。本文档命令统一改用 `configs/exp.yaml --exp <name>` 写法；打通扁平配置留作后续独立改动。



#### 文件 1: `core/pipeline/single.py`
**目标**: 集成样本统计到训练/评估循环

```python
# 在文件开头添加导入
try:
    from utils.sample_statistics import SampleStatistics
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

def seq_eval(loader, model, device, work_dir, ...):
    """评估函数"""
    # 初始化统计器
    if STATS_AVAILABLE:
        stats = SampleStatistics(
            total_samples=len(loader.dataset),
            experiment_name=Path(work_dir).name
        )
    
    for batch_idx, batch in enumerate(loader):
        sample_id = batch.get('name', f'sample_{batch_idx}')
        
        try:
            # 原有的评估逻辑
            output = model(batch)
            
            # 记录成功
            if STATS_AVAILABLE:
                stats.record_success(sample_id)
                
        except FileNotFoundError as e:
            # 文件缺失：跳过
            if STATS_AVAILABLE:
                stats.record_skip(sample_id, reason="file_not_found")
            continue
            
        except Exception as e:
            # 其他错误：记录但继续
            if STATS_AVAILABLE:
                stats.record_failure(sample_id, e)
            continue
    
    # 评估结束后保存统计
    if STATS_AVAILABLE:
        stats.save(Path(work_dir) / "sample_statistics.json")
        stats.print_summary()
        
        # 检查有效性
        if not stats.is_valid:
            print("⚠️  WARNING: Skip rate > 5%, experiment marked INVALID")
    
    return wer, stats if STATS_AVAILABLE else None
```

#### 文件 2: `core/manager/dataloader_manager.py`
**目标**: 添加 worker 初始化函数

```python
# 在文件开头添加
try:
    from utils.seed_utils import seed_worker
    SEED_WORKER_AVAILABLE = True
except ImportError:
    SEED_WORKER_AVAILABLE = False

# 在创建 DataLoader 的地方修改
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker if SEED_WORKER_AVAILABLE else None,
    prefetch_factor=prefetch_factor,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    ...
)
```

#### 文件 3: `core/manager/evaluation_manager.py`
**目标**: 集成结果保存

```python
# 在文件开头添加
try:
    from utils.result_aggregator import ExperimentResult, save_experiment_result
    RESULT_AGG_AVAILABLE = True
except ImportError:
    RESULT_AGG_AVAILABLE = False

# 在评估结束时添加
def save_evaluation_results(wer, stats, config, work_dir):
    """保存结构化的评估结果"""
    if not RESULT_AGG_AVAILABLE:
        return
    
    result = ExperimentResult(
        experiment_name=config.get('experiment_name', 'unnamed'),
        protocol=config.get('protocol', 'unified'),
        seed=config.get('random_seed', 0),
        dataset=config['dataset'],
        split='test',
        model=config['model'],
        decoder=config.get('decoder_args', {}).get('decode_mode', 'greedy'),
        wer=wer,
        total_samples=stats.total_samples if stats else None,
        successful_samples=stats.num_successful if stats else None,
        skipped_samples=stats.num_skipped if stats else None,
        failed_samples=stats.num_failed if stats else None,
        skip_rate=stats.skip_rate if stats else None,
        status=stats.status if stats else 'unknown',
        timestamp=datetime.now().isoformat(),
        work_dir=str(work_dir)
    )
    
    save_experiment_result(result, Path(work_dir) / "experiment_result.json")
```

### 1.2 数据加速状态检查

#### 已完成的加速功能（已在 main 分支）
- ✅ **VideoDataset 优化**
  - 输入索引、标签、JPEG 文件列表和 feature 缓存
  - memmap 句柄按 dataset/split/layout 复用
  - memmap 样本保留连续帧数组（避免 np.split）
  - JPEG 样本一次性 np.stack（减少复制）
  - 支持 `precache_file_lists` / `cache_file_lists` / `preopen_memmap`
  - `__getstate__` 兼容 spawn worker

- ✅ **GPU 数据路径**
  - `core/libs/gpu_video_augmentation.py`（batch crop/flip/resize/normalize）
  - `core/manager/cuda_prefetcher.py`（CUDA stream 预取 + non-blocking H2D）
  - CUDA 不可用时自动回退

- ✅ **DataLoader 配置**
  - 训练/评估分开配置 worker（eval 默认 0）
  - 支持 `prefetch_factor` / `persistent_workers` / `pin_memory`
  - worker 内限制线程数
  - `CollectManager` 预分配 padded batch
  - 可选 `LengthBucketBatchSampler`（默认关闭）

- ✅ **配置参数**（已添加到统一配置模板）
  ```yaml
  # 实验级
  eval_num_worker: 0
  prefetch_factor: 2
  persistent_workers: true
  pin_memory: true
  worker_threads: 1
  preopen_memmap: true
  gpu_prefetch: true
  length_bucket_size: 0
  
  # feeder 级
  cache_file_lists: true
  cache_features: true
  precache_file_lists: false
  gpu_augment: true
  ```

#### 待服务器验证的加速功能（未测试）
根据 `HANDOFF_DATA_LOADING_ACCELERATION.md`，以下功能已实现但未在 GPU 环境验证：

1. ⏳ **训练或 1-epoch 冒烟**
2. ⏳ **真实数据 DataLoader 读取**（memmap/JPEG 双路径）
3. ⏳ **GPU augment 运行**
4. ⏳ **CUDA prefetch benchmark**
5. ⏳ **服务器 IO 吞吐和多实验并发压测**

### 1.3 服务器验证与加速调优（4-8 小时）

按照交接文档的建议顺序执行：

```bash
# === 阶段 1: 基础验证 ===
# 1. 安装验证
bash scripts/verify_installation.sh

# 2. 配置验证
python scripts/check_protocol_compliance.py core/configs/unified_phoenix2014.yaml

# 3. 冒烟测试（1 epoch，默认配置）
cd core
python main.py \
    --config configs/exp.yaml --exp baseline \
    --num_epoch 1 \
    --work-dir /tmp/smoke_test

# 观察：GPU 利用率、H2D 传输、共享盘吞吐、worker 内存

# === 阶段 2: Worker 配置优化 ===
# 测试不同 worker 数量对性能的影响
for workers in 0 2 4 8; do
    python main.py --config configs/exp.yaml --exp baseline \
        --num-worker $workers \
        --num_epoch 1 \
        --work-dir /tmp/benchmark_w${workers}
done

# 分析每个配置的：
# - GPU 利用率
# - 训练吞吐（samples/sec）
# - 内存占用
# - IO 等待时间

# === 阶段 3: 数据类型测试 ===
# 注意:命令行 --feeder-args 是“整体覆盖”而非“合并”,必须传完整字典;
# 只传 {"datatype": "video"} 会让其它 feeder 参数退回 parser 默认值。
# 也可以选择直接改 configs/exp.yaml 里对应实验节的 feeder_args。
FEEDER_COMMON='"mode": "train", "num_gloss": -1, "drop_ratio": 1.0, "frame_interval": 1, "image_scale": 1.0, "cache_file_lists": true, "cache_features": true, "precache_file_lists": false'

# 测试 memmap vs JPEG
python main.py --config configs/exp.yaml --exp baseline \
    --feeder-args "{$FEEDER_COMMON, \"datatype\": \"memmap\", \"gpu_augment\": true}" \
    --num_epoch 1 \
    --work-dir /tmp/test_memmap

python main.py --config configs/exp.yaml --exp baseline \
    --feeder-args "{$FEEDER_COMMON, \"datatype\": \"video\", \"gpu_augment\": true}" \
    --num_epoch 1 \
    --work-dir /tmp/test_jpeg

# === 阶段 4: GPU 加速路径 ===
# 确认 PyTorch CUDA 与驱动匹配后再启用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
nvidia-smi  # 检查驱动版本

# 启用 GPU augment + prefetch(gpu_augment 属 feeder_args,见阶段 3 的完整字典)
python main.py --config configs/exp.yaml --exp baseline \
    --feeder-args "{$FEEDER_COMMON, \"datatype\": \"memmap\", \"gpu_augment\": true}" \
    --gpu-prefetch true \
    --num_epoch 1 \
    --work-dir /tmp/test_gpu_accel

# 对比关闭 GPU 加速的性能
python main.py --config configs/exp.yaml --exp baseline \
    --feeder-args "{$FEEDER_COMMON, \"datatype\": \"memmap\", \"gpu_augment\": false}" \
    --gpu-prefetch false \
    --num_epoch 1 \
    --work-dir /tmp/test_cpu_baseline

# === 阶段 5: Length Bucketing（可选）===
# 仅在不需要严格复现采样顺序时使用
python main.py --config configs/exp.yaml --exp baseline \
    --length-bucket-size 4 \
    --num_epoch 1 \
    --work-dir /tmp/test_bucket_4

python main.py --config configs/exp.yaml --exp baseline \
    --length-bucket-size 8 \
    --num_epoch 1 \
    --work-dir /tmp/test_bucket_8

# === 阶段 6: 多实验并发压测 ===
# 同时启动 2-3 个训练进程，观察：
# - 总 IO 吞吐
# - GPU 资源竞争
# - 共享盘压力
# 如果共享盘压力高，考虑数据落本地 NVMe
```

#### 优化决策树

根据验证结果调整配置：

1. **共享盘压力高**
   - 数据拷贝到本地 NVMe
   - 控制 worker 总量（所有实验加起来）
   - 使用 `cache_file_lists: true`

2. **GPU 利用率低（< 80%）**
   - 增加 `num_worker`（试 4 → 8）
   - 启用 `persistent_workers: true`
   - 启用 `gpu_prefetch: true`

3. **内存占用过高**
   - 减少 `num_worker`
   - 减少 `prefetch_factor`
   - 使用 memmap 而非加载全部到内存

4. **Padding 浪费严重**（变长视频多）
   - 启用 `length_bucket_size: 4` 或 `8`
   - **注意**：会改变采样顺序，影响复现性

5. **严格复现要求**
   - 保持 `length_bucket_size: 0`
   - 固定 `num_worker`
   - 记录最终使用的配置到实验报告

---

## 二、Agent Harness 开发（2-3 天本地开发）

### 2.1 已完成模块（70%）
- ✅ `agent/README.md` - 完整设计文档
- ✅ `agent/env.py` - OpenCSLR 环境包装器
- ✅ `agent/deepseek_client.py` - DeepSeek API 客户端
- ✅ `agent/validator.py` - 动作验证器
- ✅ `agent/budget.py` - 预算守卫
- ✅ `agent/memory.py` - 实验记忆

### 2.2 待实现模块

#### `agent/search.py` - Tier 1 解码器搜索（4-6 小时）
```python
class Tier1Search:
    """Tier 1: 解码器超参搜索（零 GPU）"""
    
    def __init__(self, env, client, memory, budget_guard):
        self.env = env
        self.client = client
        self.memory = memory
        self.budget = budget_guard
        
    def search(self, checkpoint, dataset, split="dev", 
              max_iterations=10, baseline_wer=None):
        """
        LLM-guided 解码器搜索
        
        搜索空间:
        - beam_size: [5, 10, 15, 20]
        - lm_weight: [0, 0.3, 0.6, 0.9, 1.2]
        """
        # 实现 AIDE 风格的树搜索
        pass
```

#### `agent/matrix.py` - Tier 2 实验矩阵生成（4-6 小时）
```python
class Tier2Matrix:
    """Tier 2: 实验矩阵生成"""
    
    def generate_matrix(self, research_question, base_config, dimensions):
        """
        为研究问题生成实验矩阵
        
        Example:
        - RQ1: temporal_module = [bilstm, temporalconv, transformer]
        - RQ2: model = [slowfast, tlp, vac] × dataset = [phoenix14, csl-daily]
        """
        pass
```

#### `agent/baseline.py` - 对照实验（6-8 小时）
```python
class BaselineSearch:
    """对照组: 随机搜索、网格搜索、贝叶斯优化"""
    
    def random_search(self, search_space, n_trials):
        """随机搜索"""
        pass
    
    def grid_search(self, search_space):
        """网格搜索"""
        pass
    
    def bayesian_optimization(self, search_space, n_trials):
        """贝叶斯优化（使用 Optuna）"""
        pass
```

### 2.3 实施时间线
- **Day 11-15**: 开发 Tier 1（本地，无需 GPU）
- **Day 16-18**: 开发 Tier 2 + 对照实验
- **Day 19-22**: 集成测试和调试
- **Day 23-26**: 在服务器上运行实验，收集数据
- **Day 27-30**: 分析结果，准备论文素材

---

## 三、论文集成（Agent Harness）

### 3.1 章节结构
**Section 4: Extensibility and Automation**

4.1 Registry-based Architecture  
4.2 Configuration-driven Design  
4.3 Agent-based Experiment Automation  
    - 4.3.1 Tier 1: Decoder Hyperparameter Search  
    - 4.3.2 Tier 2: Experiment Matrix Generation  
    - 4.3.3 Comparison with Baseline Methods  

### 3.2 必需实验
1. **Tier 1 效果**: 在 3-5 个 checkpoint 上搜索解码器超参
   - 报告: 最优配置、WER 改进、API 成本、迭代次数
   
2. **对照组对比**: 
   - 随机搜索 vs 网格搜索 vs 贝叶斯优化 vs LLM-guided
   - 指标: 找到最优解的速度、探索效率、最终 WER
   
3. **成本分析**:
   - API token 使用量和美元成本
   - 每 WER 点改进的成本
   - 与人工调参的时间对比（估算）

### 3.3 论文表格示例

**Table X: Decoder Search Results (Tier 1)**

| Checkpoint | Method | Iterations | Best WER | Δ WER | API Cost | Time |
|------------|--------|------------|----------|-------|----------|------|
| VAC-Phoenix14 | Random | 10 | 19.5 | -0.4 | $0 | 24min |
| VAC-Phoenix14 | Grid | 20 | 19.3 | -0.6 | $0 | 48min |
| VAC-Phoenix14 | Bayesian | 10 | 19.4 | -0.5 | $0 | 26min |
| VAC-Phoenix14 | **LLM-guided** | **7** | **19.2** | **-0.7** | **$0.38** | **18min** |

**Table Y: Search Efficiency Comparison**

| Method | Avg. Iterations to Best | Exploration Coverage | Cost per WER Point |
|--------|-------------------------|---------------------|-------------------|
| Random | 8.2 ± 2.1 | 45% | - |
| Grid | 20 (full) | 100% | - |
| Bayesian | 6.8 ± 1.4 | 62% | - |
| **LLM-guided** | **5.3 ± 1.1** | **71%** | **$0.54/point** |

---

## 四、文档清理

### 4.1 保留的核心文档
- ✅ `README.md` - 用户入口
- ✅ `INSTALL.md` - 安装指南
- ✅ `CHANGELOG.md` - 版本历史
- ✅ `docs/PROTOCOLS.md` - 实验协议规范
- ✅ `agent/README.md` - Agent 设计文档
- ✅ **本文件** - 执行计划（替代其他规划文档）

### 4.2 删除的冗余文档
- ❌ `PROGRESS.md` - 内容已整合到本文件
- ❌ `IMPLEMENTATION_SUMMARY.md` - 内容已整合到本文件
- ❌ `IMPLEMENTATION_REPORT.md` - 内容已整合到本文件
- ❌ `DEVELOPMENT_PLAN.md` - 原始规划，已执行完毕
- ❌ `OSS_REMEDIATION_SPEC.md` - 规格已内化到代码和协议文档

### 4.3 更新 CLAUDE.md
将关键信息整合到 `CLAUDE.md`，作为 AI 助手的单一入口。

---

## 五、验收清单

### 5.1 Priority 1-4（基础设施）
- [x] 安装文档和配置文件
- [x] 协议规范和工具模块
- [x] 注册表系统和验证器
- [x] 样本统计和结果汇总
- [x] 数据加速功能实现（VideoDataset + GPU 路径 + DataLoader 优化）
- [x] 代码集成完成（3 个文件，本地 CPU 测试 `core/tests/test_stats_integration.py` 通过）
- [ ] 服务器验证通过（需 GPU）
- [ ] 数据加速调优完成（确定最优配置）

### 5.2 Agent Harness
- [x] 基础模块（env, client, validator, budget, memory）
- [ ] Tier 1 搜索逻辑
- [ ] Tier 2 矩阵生成
- [ ] 对照实验实现
- [ ] 服务器实验和数据收集
- [ ] 论文素材准备

### 5.3 Priority 5-8（实验阶段）
- [ ] Phoenix2014 baseline
- [ ] Phoenix2014-T baseline
- [ ] CSL-Daily baseline
- [ ] 组件替换实验（RQ1）
- [ ] 效率分析实验（RQ2）
- [ ] 迁移实验（RQ3）
- [ ] HST 模型接入
- [ ] v1.0.0 发布

---

## 六、时间线（46 天）

| 阶段 | 日期 | 任务 | 状态 |
|------|------|------|------|
| Day 1 | 9/3-9/9 | Priority 1-4 基础设施 | ✅ 90% |
| Day 2-3 | 9/10-9/11 | 代码集成 ✅ + 服务器验证 | ⏳ 进行中 |
| Day 4-10 | 9/12-9/18 | 数据加速验证与调优 | ⏳ 待开始 |
| Day 11-22 | 9/19-9/30 | Priority 5 baseline + Agent Tier 1 | ⏳ 待开始 |
| Day 23-32 | 10/1-10/10 | Priority 6 实验矩阵 + Agent Tier 2 | ⏳ 待开始 |
| Day 33-42 | 10/11-10/20 | Priority 7-8 + 论文写作 | ⏳ 待开始 |
| Day 43-46 | 10/21-10/24 | 最终检查和提交 | ⏳ 待开始 |

---

## 七、下一步行动

### 立即执行（今天）
1. ~~完成 3 个文件的代码集成~~ ✅ 已完成（见 §1.1）
2. ~~运行本地测试验证导入无错误~~ ✅ `core/tests/test_stats_integration.py`（需完整环境时用 GPU 服务器上的 conda 环境）
3. 删除冗余文档，更新 CLAUDE.md

### 明天
1. 在服务器上运行 `verify_installation.sh`
2. 执行冒烟测试
3. 调优数据加载配置

### 本周内
1. 开始 Tier 1 搜索逻辑开发
2. 准备 Agent 测试用 mock 数据
3. 完成对照实验的基础实现

---

**关键成功因素**:
1. 代码集成必须保持向后兼容
2. Agent 不能阻塞主线实验
3. 所有工具都要有 fallback 和错误处理
4. 论文实验数据必须可复现

**联系**: 如有问题，参考 `agent/README.md` 或打开 GitHub Issue
