# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenSLR is a modular Continuous Sign Language Recognition (CSLR) toolbox built on PyTorch. It supports multiple model architectures (SlowFast, TLP, VAC, CorrNet, SEN) and datasets (Phoenix2014, Phoenix2014-T, CSL, CSL-Daily).

## Common Commands

### Training
```bash
cd core
python main.py --config configs/baseline.yaml --work-dir ./work_dir/my_experiment
```

### Testing / Evaluation
```bash
python main.py --config configs/baseline.yaml --phase test --load-weights ./work_dir/my_experiment/best_model.pt
```

### Data Preprocessing
```bash
cd core/preprocess
python dataset_preprocess.py --dataset phoenix2014 --dataset-root /path/to/dataset --process-image
```

### Adding a New Model
Models are built via factory functions in `core/models/build_function.py`. Each function (e.g., `build_slowfast`, `build_tlp`) instantiates a `SignLanguageModel` with four containers. Then reference it in a config:
```yaml
model: models.build_function.build_your_model
```

## Architecture

### Initialization Chain (core/main.py)
The system initializes via a strict chain of static managers (all in `core/manager/`):
1. `ArgumentManager` — parses CLI args + merges YAML config
2. `ConfigManager` — loads the experiment YAML
3. `ArgumentManager.map()` — applies config values to CLI parser defaults
4. `LogManager` → `DatasetManager` → `CollectManager` → `ModuleManager` → `DataloaderManager` → `DeviceManager` → `ExperimentManager`

### Model Architecture (core/models/base.py)
All models inherit from `SignLanguageModel`, which is composed of four `Container` sub-modules executed in order:
1. `spatial_module_container` — processes individual frames (e.g., ResNet, SlowFast backbone)
2. `temporal_module_container` — models temporal dependencies (e.g., TemporalConv1D, BiLSTM)
3. `loss_module_container` — computes losses (CTC-based)
4. `decoder` — converts model output to readable text (greedy max or beam search)

Data flows as a dict through all containers — each container's forward pass updates the dict in-place with new keys.

### Training Loop (core/pipeline/single.py)
`seq_train()` and `seq_eval()` contain the per-epoch logic. `ExperimentManager.run_train()` orchestrates epochs, calling `seq_train` + `seq_eval`, tracking best WER, and saving checkpoints.

### Configuration System
- Experiment configs live in `core/configs/` (e.g., `baseline.yaml`)
- Dataset-specific configs (paths, gloss dict location) also in `core/configs/` (e.g., `phoenix2014.yaml`)
- `ArgumentManager` sets dataset config at runtime based on `--dataset` arg
- `DatasetManager` loads the gloss dictionary and creates dataset instances for `train`, `dev`, `test` splits

### Data Loading
`VideoDataset` in `core/dataset/dataloader_video.py` supports multiple data types: `video` (raw jpg), `lmdb`, `memmap`, or pre-extracted features. Data type is set via `feeder_args.datatype` in the config.

### GPU Configuration
`DeviceManager` handles multi-GPU setup. Multi-GPU DataParallel is applied to `spatial_module_container` only (see `ExperimentManager.model_to_device`).
