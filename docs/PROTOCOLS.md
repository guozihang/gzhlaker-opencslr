# 实验约定 (Experimental Conventions)

本文记录 OpenCSLR 中**实际生效**的实验约定,用于保证跨模型结果可比。

**没有 protocol 开关。** 配置中不存在 `protocol` 字段,也没有 `official` /
`unified` 两套模式——统一性是默认且唯一的行为。复现论文原始设置时,在同一套
约定下按需调整该实验节的 `feeder_args` / `model_args`,并在结果表中注明差异。

## 1. 随机种子

- **配置项**:`random_seed`(默认 `0`),可用 `--random-seed` 覆盖
- **实现**:`utils/seed_utils.py` 的 `set_seed(seed, rank=0, deterministic=True)`,
  统一设置 Python `random`、NumPy、PyTorch CPU 与 CUDA 的随机状态
- **DDP**:按 `seed + rank` 派生各进程种子,保证不同进程不同但确定的随机序列
- **cuDNN**:`deterministic=True` 时启用确定性模式并关闭 benchmark
- **DataLoader**:worker 通过 `seed_worker` 获得确定性种子
- **断点续训**:RNG 状态随 checkpoint 一起保存/恢复

> 单种子结果用于**确定性复现**与**系统对比**,不用于估计方差、统计显著性
> 或置信区间。

## 2. 视频与预处理

- **数据形态**:由 `feeder_args.datatype` 选择 `video`(逐帧 jpg)、`lmdb`、
  `memmap` 或预抽取特征
- **memmap 帧尺寸**:`memmap_frame_shape: [256, 256, 3]`,由 `dataset.yaml` 指定
- **裁剪**:训练用 `video_augmentation.RandomCrop(224)`,评测用
  `CenterCrop(224)`(`dataset/dataloader_video.py`)
- **评测期确定性**:评测路径不含随机增强
- **可用键**:`feeder_args` 允许的键由 `ConfigManager.KNOWN_NESTED_KEYS` 界定,
  写错键会在启动前报错。常用键包括 `datatype`、`mode`、`frame_interval`、
  `image_scale`、`drop_ratio`、`cache_file_lists`、`gpu_augment`、`skip_fileids`

## 3. 词表与标签

- **来源**:每个数据集一份 `gloss_dict.npy`,路径由 `dataset.yaml` 的
  `dict_path` 指定(如 `./preprocess/phoenix2014/gloss_dict.npy`)
- **特殊符号**:CTC blank 为 index `0`
- **类别数**:`model_args.num_classes` 必须等于
  `len(gloss_dict) + 1`。该一致性由 `DatasetManager` 读表后校验,不一致直接报错
- **可比性**:同一数据集下所有模型必须使用同一份 `gloss_dict.npy`,换词表
  即作废跨模型对比

## 4. 解码

- **配置项**:`decode_mode`,写在 `configs/network.yaml` 的网络节里
  (`greedy` 或 `beam`),默认 `beam`
- **greedy**:逐步 argmax 后按 CTC 规则合并重复与 blank
- **beam**:CTC prefix beam search,beam 宽度由解码器自身参数决定
- **结果标注**:解码设置会写进 `decoder` 字段,必须随结果一起报告

## 5. WER 计算

- **实现**:`libs/pysclite`(纯 Python 的 sclite 移植),无外部二进制依赖
- **流程**:`EvaluationManager` 将预测写成 CTM、与 groundtruth STM 做 DP 对齐,
  统计替换/插入/删除
- **语料来源**:`dataset.yaml` 的 `evaluation_dir` 与 `evaluation_prefix`
  指向的 groundtruth STM(如 `./libs/slr_eval` + `phoenix2014-groundtruth`)
- **口径**:所有实验走同一个 `EvaluationManager` 与同一份 groundtruth STM

## 6. 样本有效性

每次评测记录总样本数、成功数、跳过数(数据缺失等)与失败数。

- **阈值**:`SKIP_RATE_THRESHOLD = 0.05`。跳过率 > 5% 的实验标记为 `invalid`
- **理由**:否则 WER 会因为少算了一批难样本而虚高

跳过率只在同一份样本集内可比,因此跨模型比较时应确认各实验的
`skipped_samples` / `failed_samples` 明细一致。

## 7. 结果落盘

评测结束时在 `work_dir` 写两个文件:

**`experiment_result.json`**(test) / **`experiment_result_dev.json`**(dev)
——由 `utils/experiment_result.py` 的 `ExperimentResult` 序列化:

```json
{
  "experiment_name": "...",
  "seed": 0,
  "dataset": "phoenix2014",
  "split": "test",
  "model": "slowfast",
  "decoder": "beam",
  "wer": 21.5,
  "total_samples": 629,
  "successful_samples": 598,
  "skipped_samples": 28,
  "failed_samples": 3,
  "skip_rate": 0.0445,
  "status": "valid",
  "timestamp": "...",
  "work_dir": "...",
  "config_path": "..."
}
```

**`sample_statistics_{dev,test}.json`** ——由 `SampleStatistics.save()` 写出,
含摘要与逐样本明细:

```json
{
  "experiment": "...",
  "total_samples": 629,
  "successful": 598,
  "skipped": 28,
  "failed": 3,
  "skip_rate": 0.0445,
  "status": "valid",
  "skip_reasons": {"missing_file": 28},
  "successful_samples": ["..."],
  "skipped_samples": ["..."],
  "failed_samples": ["..."],
  "skip_reasons_detail": {"missing_file": ["..."]},
  "errors": [{"sample": "...", "error": "...", "traceback": "..."}]
}
```

## 8. 结果报告要求

每条结果必须注明:

- **Seed**:`random_seed` 取值
- **Dataset**:数据集与划分(如 `phoenix2014/test`)
- **Model**:网络节名与关键 `model_args`
- **Decoder**:`decode_mode` 取值
- **Sample statistics**:total / successful / skipped / failed,以及
  `skip_rate` 与 `status`

**不要**在同一张表里混用不同网络配置或解码设置的结果,否则对比无意义。

## 9. 复现

要复现某个结果,需同时满足:

1. 相同的 `random_seed`
2. 相同的代码版本(git commit)
3. 相同的数据集与预处理产物
4. 相同的实验节(exp + network + dataset 三份配置)
5. 尽量相同的 PyTorch / CUDA / cuDNN 版本

即使固定了种子,不同 CUDA / cuDNN 版本与不同型号 GPU 仍可能带来微小差异。

## 10. 联系方式

- 问题反馈:https://github.com/immc-lab/OpenCSLR/issues
- 标签:`reproducibility`
