# OpenCSLR: A Unified Framework for Continuous Sign Language Recognition

[![Python](https://img.shields.io/badge/Python-3.7-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange)](https://github.com/immc-lab/OpenCSLR/releases)

## Overview

**OpenCSLR** is a unified, modular, and reproducible framework for continuous sign language recognition (CSLR) research. Unlike traditional toolboxes that simply aggregate models, OpenCSLR provides a **standardized experimental infrastructure** that enables:

- **Fair component comparison**: Compare backbones, temporal modules, losses, and decoders under identical data conventions
- **Efficiency-accuracy tradeoffs**: Systematic analysis of model parameters, GPU memory, training throughput, and inference speed
- **Cross-dataset transfer**: Evaluate generalization with unified interfaces across Phoenix2014, Phoenix2014-T, and CSL-Daily
- **Low-cost extensibility**: Add new models without modifying core training logic through a registry-based architecture

This framework prioritizes **deterministic reproducibility** with fixed seeds and one shared set of conventions, making it ideal for controlled experiments and ablation studies.

## Key Features

### Unified Experimental Protocol
- **Fixed seed reproducibility**: Deterministic training with unified random state across Python, NumPy, PyTorch, CUDA, and DataLoaders
- **Standardized preprocessing**: Consistent video decoding, frame sampling, resize, crop, and normalization
- **Unified evaluation**: Identical gloss vocabulary, decoder settings, and WER calculation across all models
- **No protocol switch**: 统一性是默认且唯一的行为,配置里没有 `unified` / `official` 开关

### Modular Architecture
- **Registry-based design**: Add models, backbones, temporal modules, losses, and decoders without modifying core code
- **Container system**: Four-stage pipeline (spatial → temporal → loss → decoder) with standardized I/O contracts
- **Configuration-driven**: All components selected via YAML configs, with key/type validation before training starts

### Supported Models & Datasets
- **Models**: SlowFast, TLP, VAC, CorrNet, SEN, and extensible to new architectures
- **Datasets**: Phoenix2014, Phoenix2014-T, CSL-Daily with unified gloss vocabularies
- **Multi-GPU training**: DataParallel support with efficient data loading

### Efficient Training Pipeline
- **Accelerated data loading**: Memory-mapped video, GPU augmentation, CUDA prefetching, and persistent workers
- **Error resilience**: Continue training on data errors, log skipped/failed samples, and record per-run sample statistics
- **Experiment tracking**: Weights & Biases integration, checkpoint management, and watchdog scripts

## Installation

### Quick Start (Recommended)

```bash
# Clone the repository
git clone https://github.com/immc-lab/OpenCSLR.git
cd OpenCSLR

# Create conda environment (includes all dependencies)
conda env create -f environment.yml
conda activate openslr

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Alternative: pip Installation

```bash
# Create virtual environment
python3.7 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.8.0+cu102 torchvision==0.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install -r requirements.txt
```

### System Requirements

- **Python**: 3.7 (recommended for scipy compatibility)
- **PyTorch**: ≥1.8.0
- **CUDA**: ≥10.2 (for GPU training)
- **GPU**: ≥8GB VRAM (training), ≥4GB (inference)
- **RAM**: ≥16GB
- **Disk**: ≥50GB for datasets

安装后可在仓库根目录运行自检脚本,它会检查依赖、配置文件与核心模块导入:

```bash
bash script/verify_installation.sh
```

## Quick Start

### 1. Data Preparation

Download and preprocess a dataset (Phoenix2014 example):

```bash
cd core/preprocess
python dataset_preprocess.py --dataset phoenix2014 \
    --dataset-root /path/to/phoenix2014 \
    --process-image
```

Supported datasets: `phoenix2014`, `phoenix2014t`, `csl-daily`

### 2. Training

Train a model:

```bash
cd core
python main.py \
    --config configs/exp.yaml \
    --exp baseline \
    --work-dir ./work_dir/slowfast_phoenix14 \
    --device 0,1
```

Key arguments:
- `--config`: Experiment configuration file (default `configs/exp.yaml`)
- `--exp`: Experiment section name inside `--config` (`baseline` / `tlp` / `vac` / `corrnet`)
- `--work-dir`: Output directory for checkpoints and logs
- `--device`: GPU IDs (comma-separated for multi-GPU), or `none` for CPU
- `--random-seed`: Random seed (default: 0, for reproducibility)

`--config` 指向的实验节通过 `network: <name>` 引用 `configs/network.yaml` 中的网络配置；
数据集路径与评测语料由 `--dataset` 在 `configs/dataset.yaml` 中选取。

### 3. Evaluation

Evaluate a trained model on the test set:

```bash
python main.py \
    --config configs/exp.yaml \
    --exp baseline \
    --phase test \
    --load-weights ./work_dir/slowfast_phoenix14_best_model.pt \
    --device 0
```

Results are written to `work_dir`:
- `experiment_result.json`（test）/ `experiment_result_dev.json`（dev）— WER 与样本统计
- `sample_statistics_{dev,test}.json` — 逐样本的成功 / 跳过 / 失败明细

## Experimental Conventions

为保证跨模型可比，所有主实验遵循同一套约定。这些约定由配置与代码保证，
没有单独的 "protocol" 开关——统一性是默认且唯一的行为。

- **Fixed global seed**: 由 `random_seed` 固定 Python / NumPy / PyTorch / CUDA 的随机状态
- **Standardized preprocessing**: 各实验共用同一套视频解码、抽帧、resize/crop、归一化路径
- **Unified vocabulary**: 每个数据集的 gloss→index 映射来自同一份 `gloss_dict.npy`
- **Consistent decoding**: `decode_mode`（greedy / beam）在 `network.yaml` 的网络节中统一指定
- **Identical WER calculation**: 所有实验走同一个 `EvaluationManager` 与 groundtruth STM

复现论文原始设置时，在同一套约定下按需调整该实验节的 `feeder_args` /
`model_args`，并在结果表中注明差异。

### Important Notes

1. **Single-seed policy**: 所有实验使用单个固定种子做确定性复现，结果**不用于**统计显著性检验或置信区间估计。

2. **Sample validity tracking**: 每次评估记录总样本数、成功数、跳过数（数据缺失等）与失败数。跳过率超过 5% 的实验被标记为 `invalid`——否则 WER 会因为少算了一批难样本而虚高。

3. **Result labeling**: 结果表中必须注明网络配置与解码设置，**不要**在同一张表里混用不同设置的结果。

## Supported Models & Datasets

### Models

All models implemented with unified four-container architecture:

| Model | Description | Config |
|-------|-------------|--------|
| **SlowFast** | Two-pathway network with fast/slow temporal streams | `build_slowfast` |
| **TLP** | Two-Stream Lightweight Pyramid | `build_tlp` |
| **VAC** | Visual Attention Consistency with knowledge distillation | `build_vac` |
| **CorrNet** | Correlation-based spatiotemporal network | `build_corrnet` |
| **SEN** | Signed Exact Network | `build_sen` |

### Datasets

| Dataset | Language | Vocab Size | Train/Dev/Test |
|---------|----------|------------|----------------|
| **Phoenix2014** | German (DGS) | 1,296 | 5,672 / 540 / 629 |
| **Phoenix2014-T** | German (DGS) | 1,066 | 7,096 / 519 / 642 |
| **CSL-Daily** | Chinese (CSL) | 2,000 | 18,401 / 1,077 / 1,176 |




## Performance Baselines

Single-seed results with fixed seed for reproducibility:

### Phoenix2014

| Model | Dev WER (%) | Test WER (%) | Params (M) |
|-------|-------------|--------------|------------|
| VAC + SMKD | 19.9 | 21.3 | - |
| SEN | 19.9 | 19.8 | - |
| CorrNet | 20.2 | 20.6 | - |
| TLP | 20.2 | 20.8 | - |
| SlowFast | 21.8 | 21.5 | - |

**Note**: These are single-seed results intended for deterministic reproduction and system comparison, not for estimating run variance or statistical significance.


## Configuration System

OpenCSLR uses a three-tier YAML configuration system:

### Example: Training SlowFast on Phoenix2014

配置分三层:exp 配置按**实验名**分节,每个实验用 `network:` 引用 network 配置
中按**网络名**分节的网络定义;数据集则由 `--dataset` 在 `dataset.yaml` 中选取。

```yaml
# core/configs/exp.yaml —— 实验节,按实验名组织
_common_experiment: &common_experiment   # YAML 锚点,复用公共字段
    feeder: dataset.dataloader_video.VideoDataset
    phase: train
    num_epoch: 80
    batch_size: 2
    random_seed: 0
    num_worker: 8
    persistent_workers: true

baseline:
    <<: *common_experiment
    network: slowfast          # 引用 network.yaml 中的网络节
    dataset: phoenix2014       # 选择 dataset.yaml 中的数据集节
    device: 0,1
    work_dir: /path/to/work_dir
```

```yaml
# core/configs/network.yaml —— 网络节,按网络名组织
slowfast:
    model: slowfast            # models/slowfast.py 中 @register_model("slowfast") 的注册名
    decode_mode: beam          # greedy 或 beam
    model_args:
        num_classes: 1296      # 须等于该数据集 gloss_dict 词表大小 + 1(CTC blank)
        hidden_size: 1024
        c2d_type: slowfast101
    loss_weights:
        SeqCTC: 1.0
        Cu: 0.001
        Cp: 0.001
        Slow: 1.0
        Fast: 1.0
```

实验节的同名键会覆盖网络节的值,因此数据集相关的开关(如 `feeder_args`)写在实验节里。

### Configuration Validation

`ConfigManager` 在合并 exp 与 network 配置后立即校验,启动前就失败,不浪费 GPU 时间:

- 嵌套节的未知键(疑似拼写错误)与值类型
- `random_seed` 为非负整数、`optimizer_args.base_lr` 为正数
- `optimizer_args.optimizer` 在受支持列表内
- `persistent_workers` 需要 `num_worker > 0`

需要运行时信息才能判定的检查不在这里。例如 `model_args.num_classes` 与实际
`gloss_dict` 词表大小是否一致,由 `DatasetManager` 读表后自行校验。

## Extending OpenCSLR

### Adding a New Model

一个模型就是一个文件。框架（`Keys`、`Container`、`SignLanguageModel`、注册表）都在
`core/models/__init__.py` 里，通用积木在 `core/modules/`，你只需要新写一个文件。

分工原则：`core/modules/` 只放**可复用的东西**，按用途分四类，**每类目录里只放
该类相关的**——`spatio/` 放空间网络、`temporal/` 放时序网络、
`losses/` 放**最小单元损失**（`CTCLoss`、`SeqKD`）、`decoders/` 放通用 `Decoder`；
不属于任何一类的辅助积木（`Identity`、`Classifier`、`NormLinear`、`TemporalLiftPooling`）
一律进 `others/`。
**模型专有的组装**（本模型怎么组合这些单元、怎么取哪个 key）写在该模型自己的文件里，
别的模型不跟着变。

**Step 1**: 新建 `core/models/my_model.py`，定义本模型的损失，组装四个容器并注册

```python
# core/models/my_model.py
import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import BiLSTM, CTCLoss, Classifier, Decoder, ResNet, SeqKD, TemporalConv1D


class MyModelLoss(nn.Module):
    """本模型的损失:把最小单元按 loss_weights 加权组合起来。"""

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.ctc = CTCLoss()          # 最小单元,来自 modules/losses
        self.kd = SeqKD(T=8)

    def forward(self, data):
        require(data, Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL,
                Keys.FEAT_LEN, Keys.LABEL_LGT, who="MyModelLoss")
        loss, total_loss = 0, {}
        for key, weight in self.loss_weights.items():
            if key == "SeqCTC":
                total_loss["SeqCTC"] = weight * self.ctc(data[Keys.SEQUENCE_LOGITS], data)
                loss += total_loss["SeqCTC"]
            # ...
        return {Keys.LOSS: loss, Keys.TOTAL_LOSS: total_loss}


@register_model("my_model")          # 注册名即 config 中 model: 的取值
def build_my_model(args, gloss_dict, loss_weights):
    return SignLanguageModel(
        spatial_module_container=Container([ResNet(args)]),
        temporal_module_container=Container([TemporalConv1D(args), BiLSTM(args), Classifier(args)]),
        loss_module_container=Container([MyModelLoss(loss_weights)]),
        decoder=Decoder(args, gloss_dict),
    )
```

如果本模型的解码要取别的 key（像 SlowFast 那样），同样在这个文件里继承
`Decoder` 覆盖 `__call__`——见 `core/models/slowfast.py` 的 `SlowFastDecoder`。

**Step 2**: 在 `core/models/__init__.py` 末尾把它加进 import 列表

```python
from . import corrnet, my_model, sen, slowfast, tlp, vac
```

导入即触发 `@register_model`，注册表随之填充，不需要改任何工厂分支。
四个容器的契约见 `core/models/__init__.py` 的 `SignLanguageModel`：每个子模块的
`forward` 接收并返回同一个 data dict（原地更新），按"空间→时序→损失→解码"执行。

**Step 3**: Reference the registered name from a network section

```yaml
# core/configs/network.yaml —— 新增一节,exp.yaml 里用 network: my_model 引用
my_model:
  model: my_model                   # 与 @register_model("my_model") 对应
  decode_mode: beam
  model_args:
      num_classes: 1296
      # your custom args
```

**That's it!** No changes to `main.py` or training logic needed.

### Cost Tracking

When adding a new model, record:
- Files added/modified
- Lines of code added
- Config entries added
- Time spent (excluding dataset download and training wait)
- Whether core training code was modified (should be "No")

This data helps quantify the framework's extensibility.

## Research Questions Enabled

This unified framework is designed to answer:

1. **RQ1 - Component Impact**: How do different backbones, temporal modules, losses, and decoders affect CSLR performance under identical conditions?

2. **RQ2 - Efficiency Tradeoffs**: What are the tradeoffs between WER, model parameters, GPU memory, training throughput, and inference speed?

3. **RQ3 - Transfer Learning**: How do unified interfaces and component choices affect generalization across compatible datasets?

See the paper for detailed experimental results addressing these questions.

## Advanced Features

### Error-Resilient Training

OpenCSLR handles data errors gracefully:

- **Missing files**: Logged and skipped
- **Decode failures**: Recorded with error type and traceback
- **Invalid predictions**: Logged and continued
- **Config/environment errors**: Fail fast before training

All experiments generate a **sample statistics report**:
```json
{
  "total_samples": 629,
  "successful": 598,
  "skipped": 28,
  "failed": 3,
  "skip_rate": 0.044,
  "status": "valid",
  "failed_samples": ["video_001.mp4", "video_042.mp4", "video_133.mp4"],
  "error_details": [...]
}
```

Experiments with >5% skip rate are automatically marked `invalid`.

### Accelerated Data Loading

Optimizations for faster training:

- **Memory-mapped video**: Pre-indexed frame access without repeated decoding
- **GPU augmentation**: Batched crop/flip/resize/normalize on GPU (B,T,C,H,W)
- **CUDA prefetching**: Asynchronous H2D transfer with CUDA streams
- **Persistent workers**: DataLoader workers stay alive across epochs
- **Length bucketing**: Group similar-length videos to reduce padding waste (optional)

Configure in your YAML:
```yaml
num_worker: 4
prefetch_factor: 2
persistent_workers: true
gpu_prefetch: true
gpu_augment: true
preopen_memmap: true
length_bucket_size: 0  # Set to 4 or 8 to enable bucketing
```

### Experiment Tracking

Built-in Weights & Biases integration:

```bash
python main.py --config configs/exp.yaml --exp baseline --wandb
```

Tracks: loss curves, WER per epoch, GPU memory, learning rate, sample statistics, and checkpoints.

## MCP Service: Experiment Management for Agents

仓库自带一个 [MCP](https://modelcontextprotocol.io) 服务,把实验管理能力暴露成标准工具,
让**已有的智能体**(Claude Code、IDE 助手等)直接调用工具来管理本仓库的实验:
查看实验清单、预览实际生效的配置、新建实验、启动/停止训练、追踪进度、读取 WER 与 checkpoint。

仓库本身不内置智能体、不调用任何大模型 API——决策在客户端那侧做,服务只负责把仓库能力
可靠地暴露出去。

```bash
pip install -r mcp_server/requirements.txt          # 只多一个 mcp 包
claude mcp add opencslr -- python3 -m mcp_server --root "$PWD"
```

仓库根目录的 `.mcp.json` 已配好项目级服务,在仓库里打开 Claude Code 会自动发现。
完整的工具清单、环境变量与设计说明见 [mcp_server/README.md](mcp_server/README.md)。

**配置只有一个真相来源。** 工具里的配置校验不是另写一份规则,而是在子进程里跑真实的
`ArgumentManager` + `ConfigManager`,因此 `resolve_experiment` 的结论与真正启动时一致——
配置错误在占上 GPU 之前就会报出来。

## Documentation

- **实验约定**: [docs/PROTOCOLS.md](docs/PROTOCOLS.md)
- **API Reference**: `docs/source/api/`,由源码注释自动生成
- **安装自检**: `bash script/verify_installation.sh`

Build docs locally:
```bash
cd docs
make gen-api  # 从 core/ 源码注释生成 API 页
make html     # 构建 HTML 文档
```

## Project Structure

```
OpenCSLR/
├── core/                      # Main source code
│   ├── main.py                # Entry point
│   ├── manager/               # Manager components
│   │   ├── argument_manager.py
│   │   ├── config_manager.py  # 配置加载 + 校验
│   │   ├── experiment_manager.py
│   │   ├── evaluation_manager.py
│   │   ├── dataloader_manager.py
│   │   ├── cuda_prefetcher.py
│   │   └── device_manager.py
│   ├── models/                # 模型层：一个模型一个文件
│   │   ├── __init__.py        # Keys + Container/SignLanguageModel + 注册表
│   │   ├── tlp.py             # 各模型的 build_* 构建函数
│   │   ├── sen.py
│   │   ├── vac.py
│   │   ├── corrnet.py
│   │   └── slowfast.py
│   ├── modules/               # 共用积木：四类目录各放各的，辅助的统一进 others/
│   │   ├── spatio/            #   空间网络（ResNet/SENresnet/corrnet_resnet/SlowFast）
│   │   │                      #   + slowfast_modules/（vendored，整块不可拆）
│   │   ├── temporal/          #   时序网络（BiLSTM/tconv/CorrNet_TemporalConv1D 等）
│   │   ├── losses/            #   最小单元损失（CTCLoss/SeqKD）
│   │   ├── decoders/          #   解码（Decoder）
│   │   └── others/            #   辅助积木（Identity/Classifier/NormLinear）
│   │                          #   + liftpool.py（TemporalLiftPooling/Local_Weighting）
│   ├── dataset/               # Dataset loaders
│   │   └── dataloader_video.py
│   ├── libs/                  # Vendored libraries
│   │   ├── pysclite/          # Pure Python WER calculator
│   │   └── gpu_video_augmentation.py
│   ├── configs/               # Configuration files
│   │   ├── exp.yaml           # 实验配置(按实验名分节)
│   │   ├── network.yaml       # 网络配置(按网络名分节)
│   │   └── dataset.yaml       # 数据集配置
│   ├── pipeline/              # Training/evaluation loops
│   │   └── single.py
│   └── preprocess/            # Data preprocessing
│       └── dataset_preprocess.py
├── script/                    # Helper / verification scripts
│   ├── run.sh                 # Training wrapper
│   ├── train_watchdog.sh      # Auto-restart on crash
│   ├── verify_installation.sh
│   ├── dump_model_structures.py  # 重构验收：对比 state_dict 的名称/形状/共享关系
│   └── convert_legacy_weights.py # 旧 checkpoint 结构校验与转换（含测试同名 test_*.py）
├── docs/                      # Documentation
├── requirements.txt           # Pip dependencies
├── environment.yml            # Conda environment
└── README.md                  # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Follow the existing code style and add tests if applicable
4. Ensure all models still train and evaluate under the shared conventions
5. Update documentation for user-facing changes
6. Submit a pull request with a clear description

For bug reports or feature requests, open an issue with:
- Your environment (OS, Python version, PyTorch version, CUDA version)
- Steps to reproduce (for bugs)
- Expected vs. actual behavior

## Citation

If you use OpenCSLR in your research, please cite:

```bibtex
@software{openslr2024,
  title={OpenCSLR: A Unified Framework for Continuous Sign Language Recognition},
  author={Guo, Zihang and Contributors},
  year={2024},
  version={1.0.0},
  publisher={GitHub},
  url={https://github.com/immc-lab/OpenCSLR}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research supported by [Your Institution/Funding]
- Built on top of PyTorch and various open-source libraries
- Inspired by state-of-the-art CSLR research

## Related Resources

- **[Awesome Continuous Sign Language Recognition](https://github.com/guozihang/awesome-continuous-sign-language-recognition)**: Comprehensive collection of CSLR papers, datasets, and resources

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/immc-lab/OpenCSLR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/immc-lab/OpenCSLR/discussions)
- **Email**: [maintainer@example.com]

## Changelog

版本历史见 `git log`。
