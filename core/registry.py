# -*- encoding: utf-8 -*-
"""通用组件注册表系统。

为所有可配置组件提供统一的注册机制，包括：
- 模型 (MODEL)
- 骨干网络 (BACKBONE)
- 时序模块 (TEMPORAL)
- 损失函数 (LOSS)
- 解码器 (DECODER)
- 数据集 (DATASET)
- 采样器 (SAMPLER)
- 数据增强 (AUGMENTATION)
- 评估器 (EVALUATOR)

新增组件时，只需使用 @register_* 装饰器，无需修改中心分发逻辑。
"""

from typing import Dict, Callable, Any, List


class Registry:
    """通用注册表类。

    管理某一类型组件的注册、查询和实例化。
    """

    def __init__(self, name: str):
        """初始化注册表。

        Args:
            name: 注册表名称（如 "MODEL", "BACKBONE"）
        """
        self.name = name
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable = None):
        """注册组件。

        可以作为装饰器或直接调用。

        Args:
            name: 组件注册名
            func: 组件构建函数（可选，装饰器模式下为 None）

        Returns:
            装饰器函数或 None

        Example:
            >>> # 装饰器模式
            >>> @MODEL_REGISTRY.register('resnet50')
            >>> def build_resnet50(**kwargs):
            >>>     return ResNet50(**kwargs)
            >>>
            >>> # 直接调用模式
            >>> MODEL_REGISTRY.register('resnet50', build_resnet50)
        """
        def decorator(f):
            if name in self._registry:
                raise ValueError(
                    f"{self.name} '{name}' is already registered. "
                    f"Existing: {self._registry[name]}, New: {f}"
                )
            self._registry[name] = f
            return f

        if func is None:
            # 装饰器模式
            return decorator
        else:
            # 直接调用模式
            return decorator(func)

    def get(self, name: str) -> Callable:
        """获取已注册的组件构建函数。

        Args:
            name: 组件注册名

        Returns:
            组件构建函数

        Raises:
            KeyError: 组件未注册时抛出
        """
        if name not in self._registry:
            available = ', '.join(sorted(self._registry.keys()))
            raise KeyError(
                f"{self.name} '{name}' not found. "
                f"Available: [{available}]"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        """构建组件实例。

        Args:
            name: 组件注册名
            **kwargs: 传递给构建函数的参数

        Returns:
            组件实例
        """
        builder = self.get(name)
        return builder(**kwargs)

    def list_available(self) -> List[str]:
        """列出所有已注册的组件名。

        Returns:
            组件名列表（已排序）
        """
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """检查组件是否已注册。

        Args:
            name: 组件名

        Returns:
            是否已注册
        """
        return name in self._registry

    def __contains__(self, name: str) -> bool:
        """支持 'name in registry' 语法。"""
        return self.is_registered(name)

    def __len__(self) -> int:
        """返回已注册组件数量。"""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', count={len(self)})"


# ============================================================================
# 创建各类型组件的注册表实例
# ============================================================================

# 模型注册表
MODEL_REGISTRY = Registry("MODEL")

# 骨干网络注册表（ResNet, SlowFast, I3D 等）
BACKBONE_REGISTRY = Registry("BACKBONE")

# 时序模块注册表（BiLSTM, TemporalConv, Transformer 等）
TEMPORAL_REGISTRY = Registry("TEMPORAL")

# 损失函数注册表（CTC Loss, Distillation Loss 等）
LOSS_REGISTRY = Registry("LOSS")

# 解码器注册表（Greedy, Beam Search, Attention 等）
DECODER_REGISTRY = Registry("DECODER")

# 数据集注册表（Phoenix2014, CSL-Daily 等）
DATASET_REGISTRY = Registry("DATASET")

# 采样器注册表（Random, Sequential, LengthBucket 等）
SAMPLER_REGISTRY = Registry("SAMPLER")

# 数据增强注册表（RandomCrop, RandomFlip, ColorJitter 等）
AUGMENTATION_REGISTRY = Registry("AUGMENTATION")

# 评估器注册表（WER, Accuracy 等）
EVALUATOR_REGISTRY = Registry("EVALUATOR")


# ============================================================================
# 便捷装饰器函数
# ============================================================================

def register_model(name: str):
    """注册模型的装饰器。

    Example:
        >>> @register_model('slowfast')
        >>> def build_slowfast(num_classes, **kwargs):
        >>>     return SlowFastModel(num_classes=num_classes, **kwargs)
    """
    return MODEL_REGISTRY.register(name)


def register_backbone(name: str):
    """注册骨干网络的装饰器。"""
    return BACKBONE_REGISTRY.register(name)


def register_temporal(name: str):
    """注册时序模块的装饰器。"""
    return TEMPORAL_REGISTRY.register(name)


def register_loss(name: str):
    """注册损失函数的装饰器。"""
    return LOSS_REGISTRY.register(name)


def register_decoder(name: str):
    """注册解码器的装饰器。"""
    return DECODER_REGISTRY.register(name)


def register_dataset(name: str):
    """注册数据集的装饰器。"""
    return DATASET_REGISTRY.register(name)


def register_sampler(name: str):
    """注册采样器的装饰器。"""
    return SAMPLER_REGISTRY.register(name)


def register_augmentation(name: str):
    """注册数据增强的装饰器。"""
    return AUGMENTATION_REGISTRY.register(name)


def register_evaluator(name: str):
    """注册评估器的装饰器。"""
    return EVALUATOR_REGISTRY.register(name)


# ============================================================================
# 全局查询函数
# ============================================================================

def list_all_registries() -> Dict[str, List[str]]:
    """列出所有注册表及其已注册组件。

    Returns:
        字典，键为注册表名，值为组件名列表
    """
    registries = {
        'MODEL': MODEL_REGISTRY,
        'BACKBONE': BACKBONE_REGISTRY,
        'TEMPORAL': TEMPORAL_REGISTRY,
        'LOSS': LOSS_REGISTRY,
        'DECODER': DECODER_REGISTRY,
        'DATASET': DATASET_REGISTRY,
        'SAMPLER': SAMPLER_REGISTRY,
        'AUGMENTATION': AUGMENTATION_REGISTRY,
        'EVALUATOR': EVALUATOR_REGISTRY,
    }

    return {
        name: registry.list_available()
        for name, registry in registries.items()
    }


def print_all_registries():
    """打印所有注册表的统计信息。"""
    all_registries = list_all_registries()

    print("\n" + "=" * 60)
    print("Registered Components")
    print("=" * 60)

    for registry_name, components in all_registries.items():
        print(f"\n{registry_name} ({len(components)}):")
        if components:
            for comp in components:
                print(f"  - {comp}")
        else:
            print("  (none)")

    print("=" * 60 + "\n")


def validate_config_references(config: Dict[str, Any]) -> List[str]:
    """验证配置中引用的所有组件是否已注册。

    Args:
        config: 配置字典

    Returns:
        错误信息列表（空列表表示验证通过）

    Example:
        >>> config = {'model': 'slowfast', 'backbone': 'resnet50'}
        >>> errors = validate_config_references(config)
        >>> if errors:
        >>>     print("Invalid config:", errors)
    """
    errors = []

    # 定义配置键到注册表的映射
    config_to_registry = {
        'model': MODEL_REGISTRY,
        'backbone': BACKBONE_REGISTRY,
        'temporal_module': TEMPORAL_REGISTRY,
        'loss': LOSS_REGISTRY,
        'decoder': DECODER_REGISTRY,
        'dataset': DATASET_REGISTRY,
        'sampler': SAMPLER_REGISTRY,
    }

    for config_key, registry in config_to_registry.items():
        if config_key in config:
            component_name = config[config_key]
            if not registry.is_registered(component_name):
                available = registry.list_available()
                errors.append(
                    f"Invalid {config_key}: '{component_name}'. "
                    f"Available: {available}"
                )

    return errors


# ============================================================================
# 向后兼容：保留旧的 MODEL_BUILDERS 接口
# ============================================================================

# 为了兼容现有代码，提供旧接口
MODEL_BUILDERS = MODEL_REGISTRY._registry


def get_model_builder(name: str) -> Callable:
    """向后兼容函数：获取模型构建函数。

    Args:
        name: 模型注册名

    Returns:
        模型构建函数
    """
    return MODEL_REGISTRY.get(name)
