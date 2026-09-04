OpenSLR: An Open-Source Toolbox for Continuous Sign Language Recognition

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Overview

OpenSLR (Open Sign Language Recognition) is a comprehensive, modular toolbox for continuous sign language recognition (CSLR) built on PyTorch. It provides a flexible and extensible architecture, making it suitable for both research and production use cases.

## Key Features

- **Modular Design**: Highly decoupled components with a manager-based architecture for easy extension and maintenance
- **Multiple Model Architectures**: Support for SlowFast, TLP (Two-Stream Lightweight Pyramid), VAC (Visual Attention Consistency), CorrNet, and other advanced models
- **Efficient Training**: Mixed precision training, memory-mapped data loading, and multi-GPU support
- **Professional Evaluation**: Word Error Rate (WER) metrics, Beam Search decoding, and detailed result analysis
- **Experiment Management**: Weights & Biases integration for experiment tracking and visualization
- **Multiple Dataset Support**: Phoenix2014, Phoenix2014-T, CSL, CSL-Daily, and customizable dataset support

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

### Prerequisites

- Python 3.7+ (Python 3.7 is recommended because `core/libs/video_augmentation.py` still uses `scipy.misc.imresize/imrotate`, which were removed in scipy 1.3.0)
- PyTorch 1.8+
- CUDA 10.2+ (for GPU acceleration)

### 1. Clone the repository

```bash
git clone https://github.com/immc-lab/OpenSLR.git
cd OpenSLR
```

### 2. Create a virtual environment (recommended)

```bash
python3.7 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you prefer to install PyTorch manually, use the official command matching your CUDA version, e.g. for CUDA 10.2:

```bash
pip install torch==1.8.0 torchvision==0.9.0
```

### 4. Quick sanity check

```bash
cd core
python main.py --config configs/exp.yaml --exp baseline --work-dir ./work_dir/smoke_test --num_epoch 1
```

## Quick Start

### 1. Data Preparation

OpenSLR supports several public sign language datasets. For example, to use the Phoenix2014 dataset:

```bash
# Download and preprocess the Phoenix2014 dataset
cd core/preprocess
python dataset_preprocess.py --dataset phoenix2014 --dataset-root /path/to/phoenix2014
```

### 2. Configure Training

Create or modify a configuration file in `core/configs/`:

```yaml
feeder: dataset.dataloader_video.VideoDataset
phase: train
dataset: phoenix2014
num_epoch: 80
work_dir: ./work_dir/baseline_experiment/
batch_size: 8

device: 0,1  # GPU devices
eval_interval: 1
save_interval: 5

model: models.build_function.build_slowfast
model_args:
    num_classes: 1296
    hidden_size: 1024
    c2d_type: slowfast101
    kernel_size: ['K5', "P2", 'K5', "P2"]

optimizer_args:
    optimizer: Adam
    base_lr: 0.0001
    step: [40, 60]
    weight_decay: 0.0001
```

### 3. Start Training

```bash
cd core
python main.py --config configs/exp.yaml --exp baseline --work-dir ./work_dir/baseline_experiment
```

## Model Architectures
- **SlowFast**: Two-pathway network for video recognition with different temporal resolutions
- **TLP (Two-Stream Lightweight Pyramid)**: Efficient architecture with spatial and temporal streams
- **VAC (Visual Attention Consistency)**: Incorporates attention mechanisms for improved performance
- **CorrNet**: Correlation-based network for sign language recognition
- **Custom Models**: Easily extendable to add new model architectures

## Datasets

OpenSLR supports several public sign language datasets:

- **Phoenix2014**: Large-scale continuous sign language recognition dataset
- **Phoenix2014-T**: German sign language dataset with temporal annotations
- **CSL**: Chinese Sign Language dataset
- **CSL-Daily**: Daily Chinese Sign Language dataset

## Evaluation

The framework provides comprehensive evaluation metrics:

```bash
# Evaluate a trained model
python main.py --config configs/exp.yaml --exp baseline --phase test --load-weights ./work_dir/baseline_experiment/best_model.pt
```




## Performance on Phoenix14

|model                        | dev wer          |test wer   |  
|-----------------------------|------------------|-----------|    
| VAC + SMKD                  | 19.9 ↑           | 21.3 ↑    |  
| TLP                         | 20.2 ↑           | 20.8 -    |  
| SEN                         | 19.9 ↑           | 19.8 ↓    |  
| CorrNet                     | 20.2 ↑           | 20.6 ↑    |  
| SlowFast                    | 21.8 ↑           | 21.5 ↑    |  


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Documentation（文档）

文档位于 `docs/` 目录,基于 Sphinx 构建。API 参考(`docs/source/api/`)由
`docs/tools/gen_api_docs.py` **从源码注释与函数签名自动生成**,无需手写,改完源码注释后一条命令重新生成:

```bash
cd docs
make gen-api   # 重新生成 docs/source/api/ 与 api.rst(幂等,可重复执行)
make html      # 构建 HTML 文档,输出在 docs/build/html/
```

手写页面(Quickstart、installation、OpenSLR 系列)不受 `make gen-api` 影响,正常维护即可。

## Resources

- **[Awesome Continuous Sign Language Recognition](https://github.com/guozihang/awesome-continuous-sign-language-recognition)**: A comprehensive collection of papers, datasets, and resources for continuous sign language recognition research

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is inspired by various state-of-the-art sign language recognition research
- We thank the contributors to the open-source libraries used in this project

## Citation

If you use this framework in your research, please cite:

```
@misc{openslr2024,
  title={OpenSLR: An Open Sign Language Recognition Framework},
  author={Your Name and Collaborators},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/yourusername/OpenSLR}},
}
```

## Contact

For questions or support, please open an issue on GitHub or contact the project maintainers.
