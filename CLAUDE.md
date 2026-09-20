# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenSLR is a modular Continuous Sign Language Recognition (CSLR) toolbox built on PyTorch. It supports multiple model architectures (SlowFast, TLP, VAC, CorrNet, SEN) and datasets (Phoenix2014, Phoenix2014-T, CSL, CSL-Daily).

## Common Commands

### Training
```bash
cd core
python main.py --config configs/exp.yaml --exp baseline --work-dir ./work_dir/my_experiment
```
`--exp` selects a section of `configs/exp.yaml`; that section's `network:` key selects a section of `configs/network.yaml`, and `dataset:` selects a section of `configs/dataset.yaml`. These three files are the only config entry points — always start from `configs/exp.yaml` with `--exp <name>`.

### Testing / Evaluation
```bash
python main.py --config configs/exp.yaml --exp baseline --phase test \
  --load-weights ./work_dir/my_experiment_best_model.pt
```
Note the checkpoint path: `work_dir` is used as a *filename prefix* for checkpoints (`{work_dir}_best_model.pt`), but as a *directory* for logs and `experiment_result.json`.

### Data Preprocessing
```bash
cd core/preprocess
python dataset_preprocess.py --dataset phoenix2014 --dataset-root /path/to/dataset --process-image
```

### Adding a New Model
One model = one file. The framework (`Keys`, `Container`, `SignLanguageModel`, the registry) lives in `core/models/__init__.py`; shared building blocks live in `core/modules/`, split by role into `spatio/`, `temporal/`, `losses/`, `decoders/`, with auxiliary blocks (`Identity`, `Classifier`, `NormLinear`) in `others/`. Each category folder holds **only** what belongs to that category — anything that isn't a spatial net / temporal net / loss / decoder goes in `others/`. `core/modules/__init__.py` re-exports all of them, so import from `modules` directly — not from the subfolders.

Add `core/models/your_model.py` with a `build_*` function that instantiates a `SignLanguageModel` from four containers and is registered via `@register_model("your_model")`, then append it to the `from . import ...` list at the bottom of `core/models/__init__.py` (importing populates the registry). Reference that name from a network section:
```yaml
# core/configs/network.yaml
your_model:
  model: your_model          # 与 @register_model("your_model") 的注册名一致
  model_args: {...}
```
Then in `core/configs/exp.yaml` add an experiment with `network: your_model`. `ModuleManager.load()` resolves only registered names — there is no dotted-path fallback.

## Architecture

### Initialization Chain (core/main.py)
The system initializes via a strict chain of static managers (all in `core/manager/`):
1. `ArgumentManager` — parses CLI args
2. `ConfigManager` — loads the experiment YAML (exp section + referenced network section) and validates it (unknown keys, types, invalid values); raises on error
3. `ArgumentManager.map()` — applies config values to CLI parser defaults (CLI > YAML > code defaults)
4. `DeviceManager` → `LogManager` → `DatasetManager` → `CollectManager` → `ModuleManager` → `DataloaderManager` → `ExperimentManager`

Training/eval loops live in `core/pipeline/single.py` (`seq_train` / `seq_eval`).

### Model Architecture (core/models/__init__.py)
All models are built as a `SignLanguageModel`, which is composed of four `Container` sub-modules executed in order:
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

## MCP Service (mcp_server/)

`mcp_server/` exposes experiment management as MCP tools so an existing agent (Claude Code, etc.) can drive this repo without the repo embedding any agent/LLM runtime itself. No model APIs are called from this side — decisions belong to the client.

- `server.py` is the **only** module that imports `mcp`; everything else (config, runs, results) is plain Python and testable without the SDK or torch.
- `core_probe.py` runs the **real** `ArgumentManager` + `ConfigManager` in a subprocess and returns the resolved config as JSON. Never re-implement config rules (key whitelist, merge order, validation) in the MCP layer — delegate, so tool verdicts match what training actually does.
- Tools are registered through `server.tool()`, not `mcp.tool()` — it translates `McpToolError` into the SDK's `ToolError`, the only exception type whose message the SDK forwards to the caller. Raising anything else reduces a useful reason ("experiment not found") to a bare `Error executing tool <name>`.
- Runs are launched detached (`start_new_session=True`) with records plus logs under `<repo>/.mcp_runs/`; `stop_run` verifies the pid's command line before signalling so it can never kill an unrelated process.
- Tests: `python -m unittest discover -s mcp_server/tests -t .` (stdlib only; no torch/GPU needed).


