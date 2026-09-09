OpenCSLR: A Unified Framework for Continuous Sign Language Recognition

[![Python](https://img.shields.io/badge/Python-3.7-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange)](https://github.com/immc-lab/OpenCSLR/releases)

## Overview

**OpenCSLR** is a unified, modular, and reproducible framework for continuous sign language recognition (CSLR) research. Unlike traditional toolboxes that simply aggregate models, OpenCSLR provides a **standardized experimental infrastructure** that enables:

- **Fair component comparison**: Compare backbones, temporal modules, losses, and decoders under identical data protocols
- **Efficiency-accuracy tradeoffs**: Systematic analysis of model parameters, GPU memory, training throughput, and inference speed
- **Cross-dataset transfer**: Evaluate generalization with unified interfaces across Phoenix2014, Phoenix2014-T, and CSL-Daily
- **Low-cost extensibility**: Add new models without modifying core training logic through a registry-based architecture

This framework prioritizes **deterministic reproducibility** with fixed seeds and unified protocols, making it ideal for controlled experiments and ablation studies.

## Key Features

### Unified Experimental Protocol
- **Fixed seed reproducibility**: Deterministic training with unified random state across Python, NumPy, PyTorch, CUDA, and DataLoaders
- **Standardized preprocessing**: Consistent video decoding, frame sampling, resize, crop, and normalization
- **Unified evaluation**: Identical gloss vocabulary, decoder settings, and WER calculation across all models
- **Two-protocol support**: `unified` for fair comparison, `official` for original paper reproduction

### Modular Architecture
- **Registry-based design**: Add models, backbones, temporal modules, losses, and decoders without modifying core code
- **Container system**: Four-stage pipeline (spatial → temporal → loss → decoder) with standardized I/O contracts
- **Configuration-driven**: All components selected via YAML configs with validation and compatibility checks

### Supported Models & Datasets
- **Models**: SlowFast, TLP, VAC, CorrNet, SEN, and extensible to new architectures
- **Datasets**: Phoenix2014, Phoenix2014-T, CSL-Daily with unified gloss vocabularies
- **Multi-GPU training**: DataParallel support with efficient data loading

### Efficient Training Pipeline
- **Accelerated data loading**: Memory-mapped video, GPU augmentation, CUDA prefetching, and persistent workers
- **Error resilience**: Continue training on data errors, log missing samples, and maintain valid sample manifests
- **Experiment tracking**: Weights & Biases integration, checkpoint management, and watchdog scripts

## Project Structure

```
OpenSLR/
├── core/                 # Main source code
│   ├── main.py           # Program entry point
│   ├── manager/          # Manager components
│   │   ├── argument_manager.py
│   │   ├── config_manager.py
│   │   ├── experiment_manager.py
│   │   └── ...
│   ├── models/           # Model architectures
│   │   ├── build_function.py
│   │   ├── modules/
│   │   └── senmodules/
│   ├── dataset/          # Dataset loaders
│   ├── libs/             # External libraries and utilities
│   ├── configs/          # Configuration files
│   └── preprocess/       # Data preprocessing scripts
└── docs/                 # Documentation
```

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

For detailed installation instructions, troubleshooting, and system-specific setup, see **[INSTALL.md](INSTALL.md)**.

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

Train a model using the unified protocol:

```bash
cd core
python main.py \
    --config configs/baseline.yaml \
    --work-dir ./work_dir/slowfast_phoenix14 \
    --dataset phoenix2014 \
    --device 0,1
```

Key arguments:
- `--config`: Experiment configuration file
- `--work-dir`: Output directory for checkpoints and logs
- `--dataset`: Dataset name (sets vocab and paths automatically)
- `--device`: GPU IDs (comma-separated for multi-GPU)
- `--seed`: Random seed (default: 0, for reproducibility)

### 3. Evaluation

Evaluate a trained model on the test set:

```bash
python main.py \
    --config configs/baseline.yaml \
    --phase test \
    --load-weights ./work_dir/slowfast_phoenix14/best_model.pt \
    --dataset phoenix2014 \
    --device 0
```

Results are saved to `work_dir/evaluation_results.json` with WER metrics and sample statistics.

## Experimental Protocols

OpenCSLR supports two experimental protocols to balance fair comparison and original reproduction:

### Unified Protocol (Default)

Used for **all main experiments** to ensure fair comparison across models:

- **Fixed global seed**: Deterministic results with unified random state (Python, NumPy, PyTorch, CUDA)
- **Standardized preprocessing**: Identical video decoding, frame sampling, resize/crop, normalization
- **Unified vocabulary**: Same gloss-to-index mapping across all models for each dataset
- **Consistent decoding**: Standardized greedy/beam search settings and text post-processing
- **Identical WER calculation**: Same evaluation script and metrics across all experiments

**Usage**: Default behavior, no special flags needed.

### Official Protocol

Preserves original paper settings for models with specialized preprocessing or decoders:

- Matches original data augmentation, sampling strategies, and label processing
- Used for reproduction verification only
- Results reported separately in supplementary materials

**Usage**: Set `protocol: official` in config or use `--protocol official`

### Important Notes

1. **Single-seed policy**: All experiments use a single fixed seed for deterministic reproduction. Results are **not** intended for statistical significance testing or confidence intervals.

2. **Sample validity tracking**: Each experiment records total samples, successful predictions, skipped samples (missing data), and failures. Experiments with >5% skipped samples are marked `invalid`.

3. **Result labeling**: All reported results must clearly indicate which protocol was used. **Never mix protocols** in the same table or comparison.

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

Results under **unified protocol** with fixed seed for reproducibility:

### Phoenix2014

| Model | Dev WER (%) | Test WER (%) | Params (M) | Protocol |
|-------|-------------|--------------|------------|----------|
| VAC + SMKD | 19.9 | 21.3 | - | unified |
| SEN | 19.9 | 19.8 | - | unified |
| CorrNet | 20.2 | 20.6 | - | unified |
| TLP | 20.2 | 20.8 | - | unified |
| SlowFast | 21.8 | 21.5 | - | unified |

**Note**: These are single-seed results intended for deterministic reproduction and system comparison, not for estimating run variance or statistical significance. Official protocol results are available in supplementary materials.  


## Configuration System

OpenCSLR uses a three-tier YAML configuration system:

### Example: Training SlowFast on Phoenix2014

```yaml
# configs/slowfast_phoenix14.yaml

# Dataset
dataset: phoenix2014
feeder: dataset.dataloader_video.VideoDataset
feeder_args:
    datatype: video  # Options: video, lmdb, memmap, features
    cache_file_lists: true
    gpu_augment: true

# Model
model: models.build_function.build_slowfast
model_args:
    num_classes: 1296
    hidden_size: 1024
    c2d_type: slowfast101

# Training
phase: train
num_epoch: 80
batch_size: 8
eval_interval: 1
save_interval: 5

# Optimization
optimizer_args:
    optimizer: Adam
    base_lr: 0.0001
    step: [40, 60]
    weight_decay: 0.0001

# Reproducibility
seed: 0
protocol: unified

# Hardware
device: 0,1
num_worker: 4
gpu_prefetch: true
```

### Configuration Validation

The framework automatically validates:
- Illegal configuration keys
- Model-dataset vocabulary compatibility
- Temporal dimension mismatches
- Decoder-model output compatibility

Errors are reported **before training starts**, saving GPU time.

## Extending OpenCSLR

### Adding a New Model

OpenCSLR's registry system allows adding models without modifying core code:

**Step 1**: Implement your model inheriting from `SignLanguageModel`

```python
# core/models/my_model.py
from core.models.base import SignLanguageModel, Container

class MyCustomModel(SignLanguageModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # Define your four containers
        self.spatial_module_container = Container(...)
        self.temporal_module_container = Container(...)
        self.loss_module_container = Container(...)
        self.decoder = Container(...)
```

**Step 2**: Register the model builder

```python
# core/models/build_function.py
from core.models.registry import register_model

@register_model
def build_my_model(num_classes, **kwargs):
    return MyCustomModel(num_classes=num_classes, **kwargs)
```

**Step 3**: Create a config file

```yaml
# configs/my_model.yaml
model: models.build_function.build_my_model
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
python main.py --config configs/baseline.yaml --wandb
```

Tracks: loss curves, WER per epoch, GPU memory, learning rate, sample statistics, and checkpoints.

## Documentation

Full documentation (coming soon):
- **Installation Guide**: [INSTALL.md](INSTALL.md)
- **Dataset Preparation**: `docs/dataset_preparation.md`
- **Training Guide**: `docs/training.md`
- **Model Extension Guide**: `docs/extending_models.md`
- **API Reference**: `docs/api/`

Build docs locally:
```bash
cd docs
make gen-api  # Generate API docs from code
make html     # Build HTML documentation
```

## Project Structure

```
OpenCSLR/
├── core/                      # Main source code
│   ├── main.py                # Entry point
│   ├── manager/               # Manager components
│   │   ├── argument_manager.py
│   │   ├── config_manager.py
│   │   ├── experiment_manager.py
│   │   ├── evaluation_manager.py
│   │   ├── dataloader_manager.py
│   │   └── device_manager.py
│   ├── models/                # Model implementations
│   │   ├── base.py            # SignLanguageModel base class
│   │   ├── registry.py        # Model registry
│   │   ├── build_function.py  # Model builders
│   │   └── modules/           # Model components
│   ├── dataset/               # Dataset loaders
│   │   ├── dataloader_video.py
│   │   └── transforms.py
│   ├── libs/                  # Utilities
│   │   ├── pysclite/          # Pure Python WER calculator
│   │   ├── gpu_video_augmentation.py
│   │   └── cuda_prefetcher.py
│   ├── configs/               # Configuration files
│   │   ├── exp.yaml           # Experiment configs
│   │   ├── network.yaml       # Model configs
│   │   └── dataset.yaml       # Dataset configs
│   ├── pipeline/              # Training/evaluation loops
│   │   └── single.py
│   └── preprocess/            # Data preprocessing
│       └── dataset_preprocess.py
├── script/                    # Helper scripts
│   ├── run.sh                 # Training wrapper
│   └── train_watchdog.sh      # Auto-restart on crash
├── docs/                      # Documentation
├── requirements.txt           # Pip dependencies
├── environment.yml            # Conda environment
├── INSTALL.md                 # Installation guide
└── README.md                  # This file
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Follow the existing code style and add tests if applicable
4. Ensure all models still work under unified protocol
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

See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.
