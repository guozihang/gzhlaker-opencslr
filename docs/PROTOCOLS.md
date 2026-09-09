# Experimental Protocols Documentation

This document specifies the experimental protocols used in OpenCSLR for reproducible and fair model comparison.

## Overview

OpenCSLR supports two experimental protocols:

1. **Unified Protocol** (default): Standardized settings for fair comparison across all models
2. **Official Protocol**: Original paper settings for reproduction verification

All results must be clearly labeled with their protocol. **Never mix protocols** in the same table or comparison.

## 1. Unified Protocol

The unified protocol ensures identical experimental conditions across all models, datasets, and runs.

### 1.1 Random Seed Management

**Fixed Global Seed**: All experiments use a single, fixed seed for deterministic reproducibility.

- **Default seed**: `0`
- **Configuration**: Set via `seed: 0` in YAML config or `--seed 0` command line
- **Scope**: Applied to all random number generators in the stack

**Implementation**:
```python
# Python built-in random
import random
random.seed(seed)

# NumPy
import numpy as np
np.random.seed(seed)

# PyTorch CPU
import torch
torch.manual_seed(seed)

# PyTorch CUDA
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # Multi-GPU

# cuDNN deterministic mode (may reduce performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**DataLoader Seeding**:
- Worker processes receive independent but deterministic seeds
- Sampler initialized with base seed
- RNG state saved/restored with checkpoints for exact resumption

**Important**: Single-seed results are for **deterministic reproduction** and **system comparison**, not for estimating run variance, statistical significance, or confidence intervals.

### 1.2 Video Preprocessing

All models use identical video preprocessing pipelines:

#### Frame Extraction
- **Format**: Individual JPEG frames or memory-mapped arrays
- **FPS**: Dataset native frame rate (25 fps for Phoenix, 30 fps for CSL-Daily)
- **Resolution**: Original resolution preserved during extraction

#### Frame Sampling
- **Strategy**: Uniform temporal sampling or dataset-specific protocol
- **Input length**: Fixed per dataset (e.g., 64 frames for SlowFast)
- **Padding**: Zero-padding or frame repetition for short videos

#### Spatial Preprocessing
- **Resize**: Shortest side to 256 pixels, preserve aspect ratio
- **Crop**: Center crop or random crop (training) to 224×224
- **Normalization**: ImageNet statistics
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Channel order**: RGB

#### Data Augmentation (Training Only)
- **Random horizontal flip**: 50% probability
- **Random crop**: 224×224 from 256×256
- **Color jittering**: Disabled by default (can enable in config)
- **Temporal augmentation**: None (preserves frame order and timing)

#### Test/Validation Preprocessing
- **Deterministic**: No randomness
- **Center crop**: 224×224
- **No flipping or color jittering

**Configuration Example**:
```yaml
feeder_args:
    datatype: video  # or memmap, lmdb, features
    resize_shape: [256, 256]
    crop_shape: [224, 224]
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    random_flip: true  # Training only
    center_crop: false  # False=random crop (train), True=center (test)
```

### 1.3 Gloss Vocabulary and Label Mapping

Each dataset has a unified gloss vocabulary:

#### Phoenix2014
- **Vocabulary size**: 1,296 glosses
- **Special tokens**: `<blank>` (CTC blank, index 0)
- **Mapping file**: `core/configs/gloss_dict/phoenix2014.txt`
- **Format**: One gloss per line, line number = gloss index

#### Phoenix2014-T
- **Vocabulary size**: 1,066 glosses
- **Special tokens**: `<blank>` (CTC blank, index 0)
- **Mapping file**: `core/configs/gloss_dict/phoenix2014t.txt`

#### CSL-Daily
- **Vocabulary size**: 2,000 glosses
- **Special tokens**: `<blank>` (CTC blank, index 0)
- **Mapping file**: `core/configs/gloss_dict/csl_daily.txt`

**Important**: All models for a given dataset must use the **same gloss vocabulary file**. Vocabulary changes invalidate cross-model comparisons.

### 1.4 Decoding Protocol

#### Greedy Decoding (Default)
- **Method**: Argmax at each time step
- **CTC collapse**: Remove consecutive duplicates and blanks
- **Post-processing**: None (raw gloss sequence)

**Implementation**:
```python
# Pseudo-code
logits = model(video)  # Shape: [B, T, vocab_size]
predictions = torch.argmax(logits, dim=-1)  # [B, T]
decoded = ctc_collapse(predictions)  # Remove duplicates and blanks
```

#### Beam Search Decoding (Optional)
- **Beam size**: 10 (configurable)
- **Length penalty**: None by default
- **Language model weight**: 0.0 (disabled by default)
- **Scorer**: CTC prefix beam search

**Configuration**:
```yaml
decoder_args:
    beam_size: 10
    lm_weight: 0.0  # Set >0 to enable language model
    length_penalty: 0.0
```

**Important**: Decoder settings must be reported with results. Changing beam size or LM weight changes performance.

### 1.5 WER Calculation

#### Metric Definition
Word Error Rate (WER) = (Substitutions + Insertions + Deletions) / Reference Length

#### Implementation
- **Tool**: Pure Python implementation (core/libs/pysclite)
- **No external dependencies**: No sclite binary, no shell calls
- **Alignment**: Levenshtein distance with standard CTC blank handling

#### Text Post-processing
- **Lowercasing**: No (preserve case from gloss vocabulary)
- **Special tokens**: Remove `<blank>`, `<eos>`, `<sos>` if present
- **Whitespace**: Strip leading/trailing, collapse multiple spaces
- **Empty predictions**: Count as errors (all deletions)

#### Per-Sample vs. Corpus-Level
- **Default**: Corpus-level WER (concatenate all hypotheses and references)
- **Optional**: Per-sample WER for error analysis

**Output Format**:
```json
{
  "wer": 21.5,
  "substitutions": 120,
  "insertions": 45,
  "deletions": 32,
  "reference_length": 917,
  "total_errors": 197
}
```

### 1.6 Training Hyperparameters

While not fully standardized (models have different optimal settings), report these for reproducibility:

- **Optimizer**: Adam (default), SGD, or other
- **Learning rate**: Initial and schedule (e.g., step decay at epochs [40, 60])
- **Weight decay**: Typically 0.0001
- **Batch size**: Per-GPU and effective (total)
- **Number of epochs**: Typically 60-80 for Phoenix, 40-60 for CSL-Daily
- **Gradient clipping**: Max norm (e.g., 5.0) if used
- **Mixed precision**: FP16 or FP32

### 1.7 Sample Validity Tracking

Every experiment must record:

```json
{
  "total_samples": 629,
  "successful": 598,
  "skipped": 28,
  "failed": 3,
  "skip_rate": 0.0445,
  "status": "valid",
  "skipped_samples": ["video_001", "video_042", ...],
  "failed_samples": ["video_133"],
  "error_details": [
    {"sample": "video_133", "error": "RuntimeError: CUDA OOM", "traceback": "..."}
  ]
}
```

**Validity Criteria**:
- `skip_rate ≤ 0.05` (5%): Experiment marked `valid`
- `skip_rate > 0.05`: Experiment marked `invalid`, excluded from main results

**Sample Manifest**: All models must evaluate on the **same set of valid samples** for fair comparison.

## 2. Official Protocol

The official protocol preserves original paper settings for models with specialized preprocessing or decoders.

### 2.1 When to Use Official Protocol

Use official protocol when:
- Original paper uses non-standard preprocessing (e.g., different resize, custom augmentation)
- Model requires specific frame sampling (e.g., non-uniform, motion-based)
- Custom gloss vocabulary or label mapping
- Specialized decoder (e.g., attention-based, not CTC)

### 2.2 Configuration

```yaml
protocol: official
official_settings:
    # Model-specific settings
    preprocessing: custom_preprocess_function
    vocabulary: path/to/custom_vocab.txt
    decoder: custom_decoder
```

### 2.3 Reporting

Official protocol results must be:
- Clearly labeled as "Official Protocol"
- Reported separately from unified protocol results
- Placed in supplementary materials or reproduction reports
- Never mixed with unified protocol in the same table

## 3. Result Reporting Requirements

Every reported result must include:

### 3.1 Mandatory Metadata
- **Protocol**: `unified` or `official`
- **Seed**: The random seed used (e.g., `0`)
- **Dataset**: Dataset name and split (e.g., `phoenix2014/test`)
- **Model**: Model name and configuration (e.g., `SlowFast-101`)
- **Decoder**: Decoder type and settings (e.g., `greedy` or `beam-10`)

### 3.2 Sample Statistics
- Total samples
- Successful predictions
- Skipped samples (with reasons)
- Failed samples (with error types)
- Skip rate and validity status

### 3.3 Performance Metrics
- WER (%)
- Substitutions, insertions, deletions (counts)
- Reference length

### 3.4 Optional Efficiency Metrics
- Model parameters (M)
- Peak GPU memory (GB)
- Training throughput (samples/sec)
- Total training time (GPU-hours)
- Inference time per video (ms)

### Example Result Entry

```json
{
  "experiment": "slowfast_phoenix14_unified",
  "protocol": "unified",
  "seed": 0,
  "dataset": "phoenix2014",
  "split": "test",
  "model": "SlowFast-101",
  "decoder": "greedy",
  "samples": {
    "total": 629,
    "successful": 598,
    "skipped": 28,
    "failed": 3,
    "skip_rate": 0.0445,
    "status": "valid"
  },
  "performance": {
    "wer": 21.5,
    "substitutions": 120,
    "insertions": 45,
    "deletions": 32,
    "reference_length": 917
  },
  "efficiency": {
    "params_m": 45.2,
    "peak_memory_gb": 8.3,
    "inference_ms_per_video": 42.1
  },
  "timestamp": "2024-09-09T10:30:00Z",
  "commit": "b011062"
}
```

## 4. Reproducibility Guidelines

### 4.1 Exact Reproduction

To exactly reproduce a result:
1. Use the **same seed**
2. Use the **same code version** (git commit)
3. Use the **same dataset** (including preprocessing)
4. Use the **same configuration file**
5. Use the **same PyTorch/CUDA versions** (if possible)
6. Use the **same sample manifest** (exclude the same failed samples)

### 4.2 Expected Variation

Even with fixed seeds, minor variations may occur due to:
- CUDA driver versions
- cuDNN versions
- Hardware differences (different GPU models)
- Operating system differences

**Typical variation**: ±0.1-0.3% WER for well-behaved models

### 4.3 Version Pinning

For strict reproducibility, pin these versions:
```
python==3.7.x
torch==1.8.0
torchvision==0.9.0
numpy==1.19.x
opencv-python==4.5.x
scipy==1.2.x
```

## 5. Protocol Validation Checklist

Before submitting results, verify:

- [ ] Protocol (`unified` or `official`) clearly stated
- [ ] Seed value recorded in config and results
- [ ] All models use same gloss vocabulary for the dataset
- [ ] Preprocessing settings match unified protocol (if using unified)
- [ ] Decoder settings reported
- [ ] Sample statistics included (total/success/skip/fail)
- [ ] Skip rate ≤ 5% (or experiment marked invalid)
- [ ] Same sample manifest used across models
- [ ] No protocol mixing in comparison tables
- [ ] Code version (git commit) recorded
- [ ] Configuration file saved with results

## 6. Protocol Evolution

This document describes the protocol for **OpenCSLR v1.0.0**. 

Future versions may introduce new protocols (e.g., `unified_v2`). When this happens:
- Old results remain valid with their protocol label
- New protocols must be backward-incompatible or clearly superior
- Protocol version must be included in all results
- Migration guide must be provided

## Contact

For protocol questions or clarifications:
- Open an issue: https://github.com/immc-lab/OpenCSLR/issues
- Label: `protocol` or `reproducibility`
