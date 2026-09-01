# -*- encoding: utf-8 -*-
"""OpenCSLR 配置管理器模块。

负责加载实验配置:从 exp 配置文件中按实验名取节,再按该实验引用的
网络名从 network.yaml 取网络配置,两者合并为全局配置数据。
"""

import os
import yaml
from .argument_manager import ArgumentManager

class ConfigManager:
    """配置管理器。

    使用类方法管理实验配置数据。配置文件拆分:
      - exp 配置文件(默认 configs/exp.yaml):按实验名分节,包含实验设置,
        各实验通过 ``network: <name>`` 引用网络配置;
      - network.yaml:与 exp 配置文件同目录,按网络名分节,包含网络相关配置。
    """

    # 各嵌套配置节允许的键集合,用于在加载时发现拼写错误。
    # 新增配置键时需同步更新对应集合。
    KNOWN_NESTED_KEYS = {
        "model_args": {
            "num_classes", "hidden_size", "c2d_type", "conv_type",
            "kernel_size", "use_bn", "share_classifier", "weight_norm",
            "slowfast_config", "stride",
        },
        "feeder_args": {
            "mode", "datatype", "num_gloss", "drop_ratio", "frame_interval",
            "image_scale", "skip_fileids", "skip_indices", "skip_info_path",
            "allowable_vid_length", "limit_len",
        },
        "optimizer_args": {
            "optimizer", "base_lr", "step", "learning_ratio", "weight_decay",
            "start_epoch", "nesterov",
        },
        "wandb": {"enable", "project", "entity"},
    }

    @classmethod
    def init(cls):
        """初始化配置管理器,加载 exp 与 network 配置并合并。"""
        experiment_config_path = ArgumentManager.get( "config" )
        cls.load_experiment(experiment_config_path)

    @classmethod
    def _validate(cls, data):
        """校验合并后的配置中嵌套节的键是否合法。

        Args:
            data: 合并后的配置数据。

        Raises:
            ValueError: 嵌套节出现未知键(疑似拼写错误)时抛出。
        """
        for section, known in cls.KNOWN_NESTED_KEYS.items():
            value = data.get(section)
            if not isinstance(value, dict):
                continue
            unknown = set(value) - known
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in '{section}': {sorted(unknown)}. "
                    f"Allowed: {sorted(known)}."
                )

    @classmethod
    def load_experiment(cls, config_path):
        """加载 exp 配置文件中的指定实验节,并合并其引用的网络配置。

        Args:
            config_path: exp 配置文件路径(按实验名分节的 YAML)。

        Raises:
            ValueError: 实验名或引用的网络名不存在时抛出。
            TypeError: 配置节内容不是字典类型时抛出。
        """
        exp_name = ArgumentManager.get("exp")
        with open(config_path, mode="r", encoding="utf-8") as f:
            exp_all = yaml.load(f, Loader=yaml.FullLoader) or {}
        if not isinstance(exp_all, dict):
            raise TypeError("仅支持字典类型的配置数据，请检查配置文件内容。")
        if exp_name not in exp_all:
            raise ValueError(
                f"Unknown experiment {exp_name!r}: not found in {config_path}. "
                f"Available: {sorted(exp_all)}."
            )
        exp_data = exp_all[exp_name]
        if not isinstance(exp_data, dict):
            raise TypeError("仅支持字典类型的配置数据，请检查配置文件内容。")

        network_name = exp_data.get("network")
        if not network_name:
            raise ValueError(f"Experiment {exp_name!r} must specify 'network' in {config_path}.")

        config_dir = os.path.dirname(os.path.abspath(config_path))
        network_path = os.path.join(config_dir, "network.yaml")
        with open(network_path, mode="r", encoding="utf-8") as f:
            network_all = yaml.load(f, Loader=yaml.FullLoader) or {}
        if not isinstance(network_all, dict):
            raise TypeError("仅支持字典类型的配置数据，请检查配置文件内容。")
        if network_name not in network_all:
            raise ValueError(
                f"Unknown network {network_name!r}: not found in {network_path}. "
                f"Available: {sorted(network_all)}."
            )
        network_data = network_all[network_name]
        if not isinstance(network_data, dict):
            raise TypeError("仅支持字典类型的配置数据，请检查配置文件内容。")

        merged = dict(network_data)
        merged.update(exp_data)  # 实验节可覆盖网络节的同名配置
        merged.pop("network", None)  # network 仅为引用字段,不属于配置参数
        cls._validate(merged)
        setattr(cls, 'CONFIG_DATA', merged)

    @classmethod
    def get(cls, key=None):
        """获取配置数据。

        Args:
            key: 配置键名。若为 None，则返回全部配置数据；否则返回指定键的值。

        Returns:
            全部配置数据字典或指定键的配置值
        """
        if key is None:
            return getattr(cls, 'CONFIG_DATA')
        else:
            return getattr(cls, 'CONFIG_DATA')[key]

    @classmethod
    def __iter__(cls):
        """迭代配置数据的键。

        Returns:
            iterator: 配置数据字典的键迭代器
        """
        return iter(getattr(cls, 'CONFIG_DATA'))
