OpenSLR: An Open-Source Toolbox for Continuous Sign Language Recognition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
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
│   │   └── ...
│   ├── dataset/          # Dataset loaders
│   ├── libs/             # External libraries and utilities
│   ├── configs/          # Configuration files
│   └── preprocess/       # Data preprocessing scripts
└── docs/                 # Documentation
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+ (for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/immc-lab/OpenSLR.git
cd OpenSLR

# Install dependencies
pip install -r requirements.txt
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
python main.py --model models.build_function.build_vac --work-dir ./work_dir/openslr/vac/14/ --dataset phoenix2014 --config ./configs/vac.yaml
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
python main.py --model models.build_function.build_vac --dataset phoenix2014 --config ./configs/vac.yaml --phase test --load-checkpoints ./work_dir/openslr/vac/14/best_model.pt
```




## Performance Comparison

<table>
  <caption><strong>Table 1:</strong> Performance comparison against originally reported results. Arrows indicate deviation from the original (&uarr; better, &darr; worse, = equal, - no official result).</caption>
  <thead>
    <tr>
      <th rowspan="3">Method</th>
      <th rowspan="3">Reference</th>
      <th colspan="4">Phoenix14</th>
      <th colspan="4">Phoenix14T</th>
      <th colspan="4">CSL-Daily</th>
    </tr>
    <tr>
      <th colspan="2">Ours</th>
      <th colspan="2">Original</th>
      <th colspan="2">Ours</th>
      <th colspan="2">Original</th>
      <th colspan="2">Ours</th>
      <th colspan="2">Original</th>
    </tr>
    <tr>
      <th>dev</th>
      <th>test</th>
      <th>dev</th>
      <th>test</th>
      <th>dev</th>
      <th>test</th>
      <th>dev</th>
      <th>test</th>
      <th>dev</th>
      <th>test</th>
      <th>dev</th>
      <th>test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VAC+SMKD</td>
      <td>ICCV'21</td>
      <td>19.2 &uarr;</td>
      <td>20.3 &uarr;</td>
      <td>19.8</td>
      <td>20.5</td>
      <td>19.9</td>
      <td>20.7</td>
      <td>-</td>
      <td>-</td>
      <td>27.6</td>
      <td>27.6</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>TLP</td>
      <td>ECCV'22</td>
      <td>20.4 &darr;</td>
      <td>20.8 =</td>
      <td>19.7</td>
      <td>20.8</td>
      <td>19.5 &darr;</td>
      <td>21.4 &darr;</td>
      <td>19.4</td>
      <td>21.2</td>
      <td>33.0</td>
      <td>31.9</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>SEN</td>
      <td>AAAI'23</td>
      <td>20.6 &darr;</td>
      <td>20.6 &uarr;</td>
      <td>19.5</td>
      <td>21.0</td>
      <td>19.9 &darr;</td>
      <td>20.6 &uarr;</td>
      <td>19.3</td>
      <td>20.7</td>
      <td>32.2 &darr;</td>
      <td>30.9 &darr;</td>
      <td>31.1</td>
      <td>30.7</td>
    </tr>
    <tr>
      <td>CorrNet</td>
      <td>CVPR'23</td>
      <td>19.5 &darr;</td>
      <td>19.4 =</td>
      <td>18.8</td>
      <td>19.4</td>
      <td>18.9 =</td>
      <td>19.9 &uarr;</td>
      <td>18.9</td>
      <td>20.5</td>
      <td>27.6 &uarr;</td>
      <td>26.4 &uarr;</td>
      <td>30.6</td>
      <td>30.1</td>
    </tr>
    <tr>
      <td>SlowFast</td>
      <td>ICASSP'24</td>
      <td>18.5 &darr;</td>
      <td>18.7 &darr;</td>
      <td>18.0</td>
      <td>18.3</td>
      <td>18.7 &darr;</td>
      <td>19.7 &darr;</td>
      <td>17.7</td>
      <td>18.7</td>
      <td>25.7 &darr;</td>
      <td>25.1 &darr;</td>
      <td>25.5</td>
      <td>24.9</td>
    </tr>
  </tbody>
</table>

## Pretrained Model Weights
Below are download links for the best model weights of 5 state-of-the-art continuous sign language recognition (CSLR) models trained on three benchmark datasets: CSL-Daily, Phoenix14 and Phoenix14-T.

### VAC + SMKD (ICCV 2021) ：
- Phoenix14:  [download from Baidu Drive](https://pan.baidu.com/s/1XVDlqIVebkLniN6ZvF3xsw?pwd=ytip)
- Phoenix14-T:  [download from Baidu Drive](https://pan.baidu.com/s/1A0N8sNUyPeZNqQ2S93jqlQ?pwd=1ryb)
- CSL-Daily: [download from Baidu Drive](https://pan.baidu.com/s/1hR_JzI7F_7zxtVFR6j-Aag?pwd=e1t6)

### TLP (ECCV 2022)：
- Phoenix14: [download from Baidu Drive](https://pan.baidu.com/s/1rxW0Bfp3q_o3ypxZMaVqSA?pwd=53gc)
- Phoenix14-T: [download from Baidu Drive](https://pan.baidu.com/s/1xbgIbLTMNqWrLyhvUaJi_Q?pwd=jdhu)
- CSL-Daily: [download from Baidu Drive](https://pan.baidu.com/s/1T0ugjdP3AKMEHQkFp__GmQ?pwd=5z53)

### SEN (AAAI 2023)
- Phoenix14: [download from Baidu Drive](https://pan.baidu.com/s/1tFelts-E7XEuibjJMz88Uw?pwd=gt98)
- Phoenix14-T: [download from Baidu Drive](https://pan.baidu.com/s/1KtK5EhUmlRkL8WEKjjMOBg?pwd=qehs)
- CSL-Daily: [download from Baidu Drive](https://pan.baidu.com/s/1zmQMFCes_Hh0sTEjc6ITKA?pwd=53tu)

### CorrNet (CVPR 2023)
- Phoenix14: [download from Baidu Drive](https://pan.baidu.com/s/1bu0hJaMfDBoCShB5USuKfw?pwd=diwp)
- Phoenix14-T: [download from Baidu Drive](https://pan.baidu.com/s/1wMtZV6XQCDrqNKtF6A1Qpg?pwd=v4si)
- CSL-Daily: [download from Baidu Drive](https://pan.baidu.com/s/16tD4ZOZ1KTgUBPJGwrYKHA?pwd=cue3)

### SlowFast (ICASSP 2024)
- Phoenix14: [download from Baidu Drive](https://pan.baidu.com/s/1c0h4dU30NZd4XytUDrGaBw?pwd=j7er)
- Phoenix14-T: [download from Baidu Drive](https://pan.baidu.com/s/1tRxC06MdtOPer0YQBy6IdA?pwd=meqd)
- CSL-Daily: [download from Baidu Drive](https://pan.baidu.com/s/1DTdIQY8zTitAoBG5KpwJ9w?pwd=ke7j)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

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
