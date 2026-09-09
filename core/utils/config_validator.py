# -*- encoding: utf-8 -*-
"""配置验证与兼容性检查工具。

在训练开始前验证配置的合法性和组件间的兼容性，
及早发现配置错误，避免浪费 GPU 时间。
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml


class ValidationError(Exception):
    """配置验证错误。"""
    pass


class ConfigValidator:
    """配置验证器。

    检查配置文件的合法性、组件兼容性、必需字段等。
    """

    # 必需的顶层配置字段
    REQUIRED_FIELDS = [
        'dataset',
        'model',
        'phase',
    ]

    # 已知的数据集及其词表大小
    DATASET_VOCAB_SIZES = {
        'phoenix2014': 1296,
        'phoenix2014t': 1066,
        'csl-daily': 2000,
    }

    # 支持的协议
    VALID_PROTOCOLS = ['unified', 'official']

    # 支持的阶段
    VALID_PHASES = ['train', 'test', 'features']

    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        """初始化验证器。

        Args:
            config: 配置字典
            verbose: 是否打印详细信息
        """
        self.config = config
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> Tuple[bool, List[str], List[str]]:
        """执行所有验证检查。

        Returns:
            (is_valid, errors, warnings) 元组
        """
        self.errors = []
        self.warnings = []

        # 执行各项检查
        self._check_required_fields()
        self._check_protocol()
        self._check_seed()
        self._check_dataset()
        self._check_model()
        self._check_optimizer()
        self._check_dataloader()
        self._check_vocab_compatibility()

        is_valid = len(self.errors) == 0

        if self.verbose:
            self._print_results()

        return is_valid, self.errors, self.warnings

    def _check_required_fields(self):
        """检查必需字段是否存在。"""
        for field in self.REQUIRED_FIELDS:
            if field not in self.config:
                self.errors.append(f"Missing required field: '{field}'")

    def _check_protocol(self):
        """检查实验协议设置。"""
        protocol = self.config.get('protocol', 'unified')

        if protocol not in self.VALID_PROTOCOLS:
            self.errors.append(
                f"Invalid protocol: '{protocol}'. "
                f"Must be one of {self.VALID_PROTOCOLS}"
            )

        # 协议字段应该明确指定
        if 'protocol' not in self.config:
            self.warnings.append(
                "Protocol not specified, using default 'unified'. "
                "Consider explicitly setting 'protocol: unified' in config."
            )

    def _check_seed(self):
        """检查随机种子设置。"""
        # 检查是否启用 random_fix
        if not self.config.get('random_fix', True):
            self.warnings.append(
                "random_fix is False. Experiments will not be reproducible. "
                "For unified protocol, set 'random_fix: true'."
            )

        # 检查 seed 值
        seed = self.config.get('random_seed', 0)
        if not isinstance(seed, int):
            self.errors.append(f"random_seed must be an integer, got {type(seed)}")

        if seed < 0:
            self.warnings.append(
                f"random_seed is negative ({seed}). "
                "Consider using non-negative seeds for clarity."
            )

        # unified 协议应该使用固定的 seed
        protocol = self.config.get('protocol', 'unified')
        if protocol == 'unified' and 'random_seed' not in self.config:
            self.warnings.append(
                "Using unified protocol but random_seed not explicitly set. "
                "Default (0) will be used. Consider explicitly setting it."
            )

    def _check_dataset(self):
        """检查数据集配置。"""
        dataset = self.config.get('dataset')

        if not dataset:
            return  # Already caught by required fields check

        # 检查是否是已知数据集
        if dataset not in self.DATASET_VOCAB_SIZES:
            self.warnings.append(
                f"Unknown dataset: '{dataset}'. "
                f"Known datasets: {list(self.DATASET_VOCAB_SIZES.keys())}"
            )

        # 检查 feeder 配置
        if 'feeder' not in self.config:
            self.warnings.append(
                "No 'feeder' specified. Default data loader will be used."
            )

    def _check_model(self):
        """检查模型配置。"""
        model = self.config.get('model')

        if not model:
            return  # Already caught by required fields check

        # 检查模型参数
        model_args = self.config.get('model_args', {})

        # num_classes 应该与数据集词表大小匹配
        if 'num_classes' in model_args:
            num_classes = model_args['num_classes']
            dataset = self.config.get('dataset')

            if dataset in self.DATASET_VOCAB_SIZES:
                expected = self.DATASET_VOCAB_SIZES[dataset]
                if num_classes != expected:
                    self.errors.append(
                        f"Model num_classes ({num_classes}) does not match "
                        f"dataset vocab size ({expected}) for {dataset}. "
                        "This will cause dimension mismatch errors."
                    )

    def _check_optimizer(self):
        """检查优化器配置。"""
        if self.config.get('phase') != 'train':
            return  # 只在训练时需要优化器

        optimizer_args = self.config.get('optimizer_args', {})

        if not optimizer_args:
            self.warnings.append(
                "No optimizer_args specified. Default settings will be used."
            )
            return

        # 检查必需的优化器参数
        if 'optimizer' not in optimizer_args:
            self.warnings.append("No optimizer type specified in optimizer_args")

        if 'base_lr' not in optimizer_args:
            self.warnings.append("No base_lr specified in optimizer_args")

        # 检查学习率合理性
        base_lr = optimizer_args.get('base_lr')
        if base_lr is not None:
            if base_lr <= 0:
                self.errors.append(f"base_lr must be positive, got {base_lr}")
            if base_lr > 1:
                self.warnings.append(
                    f"base_lr is very large ({base_lr}). "
                    "Typical values are 1e-4 to 1e-3 for Adam."
                )

    def _check_dataloader(self):
        """检查数据加载器配置。"""
        num_worker = self.config.get('num_worker', 0)
        eval_num_worker = self.config.get('eval_num_worker', 0)

        # 检查 worker 数量合理性
        if num_worker > 16:
            self.warnings.append(
                f"num_worker is very large ({num_worker}). "
                "This may cause excessive memory usage. Typical values: 2-8."
            )

        # 检查 persistent_workers 与 num_worker 的兼容性
        persistent_workers = self.config.get('persistent_workers', False)
        if persistent_workers and num_worker == 0:
            self.warnings.append(
                "persistent_workers is True but num_worker is 0. "
                "persistent_workers only works with num_worker > 0."
            )

        # 检查 GPU 加速选项
        gpu_augment = self.config.get('gpu_augment', False)
        gpu_prefetch = self.config.get('gpu_prefetch', False)

        if (gpu_augment or gpu_prefetch) and self.config.get('device') == 'cpu':
            self.warnings.append(
                "GPU augmentation/prefetching enabled but device is 'cpu'. "
                "These options require CUDA."
            )

    def _check_vocab_compatibility(self):
        """检查词表兼容性（dataset vs model vs decoder）。"""
        dataset = self.config.get('dataset')
        model_args = self.config.get('model_args', {})

        if dataset not in self.DATASET_VOCAB_SIZES:
            return  # 未知数据集，跳过

        expected_vocab_size = self.DATASET_VOCAB_SIZES[dataset]
        model_num_classes = model_args.get('num_classes')

        if model_num_classes and model_num_classes != expected_vocab_size:
            self.errors.append(
                f"Vocabulary size mismatch: "
                f"dataset '{dataset}' expects {expected_vocab_size} classes, "
                f"but model has num_classes={model_num_classes}. "
                "This will cause errors during training/inference."
            )

    def _print_results(self):
        """打印验证结果。"""
        print("\n" + "=" * 60)
        print("Configuration Validation Results")
        print("=" * 60)

        if not self.errors and not self.warnings:
            print("✓ All checks passed!")
        else:
            if self.errors:
                print(f"\n❌ {len(self.errors)} Error(s):")
                for i, error in enumerate(self.errors, 1):
                    print(f"  {i}. {error}")

            if self.warnings:
                print(f"\n⚠️  {len(self.warnings)} Warning(s):")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"  {i}. {warning}")

        print("=" * 60 + "\n")


def validate_config_file(config_path: str, verbose: bool = True) -> bool:
    """验证配置文件。

    Args:
        config_path: 配置文件路径
        verbose: 是否打印详细信息

    Returns:
        是否通过验证

    Raises:
        ValidationError: 如果验证失败
    """
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 验证
    validator = ConfigValidator(config, verbose=verbose)
    is_valid, errors, warnings = validator.validate_all()

    if not is_valid:
        error_msg = "Configuration validation failed:\n" + "\n".join(errors)
        raise ValidationError(error_msg)

    return True


def check_model_dataset_compatibility(model_name: str,
                                     dataset_name: str,
                                     num_classes: int) -> bool:
    """检查模型与数据集的兼容性。

    Args:
        model_name: 模型名称
        dataset_name: 数据集名称
        num_classes: 模型的类别数

    Returns:
        是否兼容
    """
    if dataset_name not in ConfigValidator.DATASET_VOCAB_SIZES:
        print(f"Warning: Unknown dataset '{dataset_name}'")
        return True  # 未知数据集，无法验证

    expected = ConfigValidator.DATASET_VOCAB_SIZES[dataset_name]

    if num_classes != expected:
        print(f"❌ Incompatible: {model_name} has {num_classes} classes, "
              f"but {dataset_name} requires {expected}")
        return False

    print(f"✓ Compatible: {model_name} matches {dataset_name} "
          f"vocab size ({num_classes})")
    return True


def check_temporal_dimensions(model_output_len: int,
                             expected_len: int,
                             tolerance: float = 0.1) -> bool:
    """检查时序维度兼容性。

    Args:
        model_output_len: 模型输出的时序长度
        expected_len: 期望的时序长度
        tolerance: 允许的相对误差

    Returns:
        是否兼容
    """
    if model_output_len == expected_len:
        return True

    ratio = abs(model_output_len - expected_len) / expected_len
    if ratio <= tolerance:
        print(f"⚠️  Temporal dimension close but not exact: "
              f"{model_output_len} vs {expected_len} (within {tolerance*100}% tolerance)")
        return True

    print(f"❌ Temporal dimension mismatch: "
          f"{model_output_len} vs {expected_len} (exceeds {tolerance*100}% tolerance)")
    return False


def validate_before_training(config_path: str,
                            check_files: bool = True,
                            check_gpu: bool = True) -> bool:
    """训练前的完整验证。

    Args:
        config_path: 配置文件路径
        check_files: 是否检查文件存在性
        check_gpu: 是否检查 GPU 可用性

    Returns:
        是否通过所有检查
    """
    print("Starting pre-training validation...")

    # 1. 验证配置文件
    try:
        validate_config_file(config_path, verbose=True)
    except ValidationError as e:
        print(f"Configuration validation failed: {e}")
        return False

    # 2. 检查 GPU（如果需要）
    if check_gpu:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available. Training will use CPU (very slow).")
        else:
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")

    # 3. 检查文件存在性（如果需要）
    if check_files:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 检查必要的文件
        files_to_check = []

        if 'load_weights' in config and config['load_weights']:
            files_to_check.append(('Model weights', config['load_weights']))

        if 'load_checkpoints' in config and config['load_checkpoints']:
            files_to_check.append(('Checkpoint', config['load_checkpoints']))

        missing_files = []
        for name, path in files_to_check:
            if not Path(path).exists():
                missing_files.append(f"{name}: {path}")

        if missing_files:
            print("⚠️  Warning: Some files are missing:")
            for item in missing_files:
                print(f"  - {item}")

    print("\n✅ Pre-training validation complete!")
    return True
