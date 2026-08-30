# -*- encoding: utf-8 -*-
"""OpenCSLR 配置管理器模块。

负责加载和访问实验配置数据，支持 YAML 格式的配置文件解析，
并提供全局配置数据的存储与读取接口。
"""

import json
import yaml
import toml
from .argument_manager import ArgumentManager

class ConfigManager:
    """配置管理器。

    使用类方法管理实验配置数据，支持从 YAML 配置文件加载配置，
    并通过键值对方式访问配置项。
    """

    @classmethod
    def init(cls):
        """初始化配置管理器，从 ArgumentManager 中获取配置文件路径并加载。
        """
        experiment_config_path = ArgumentManager.get( "config" )
        cls.load(experiment_config_path)

    @classmethod
    def load(cls, config_path):
        """加载 YAML 配置文件。

        读取指定路径的 YAML 配置，将其内容合并到全局配置数据 CONFIG_DATA 中。

        Args:
            config_path: 配置文件路径

        Raises:
            TypeError: 当配置文件内容不是字典类型时抛出
        """
        with open ( config_path , mode = "r" ) as f :
            data = yaml.load ( f , Loader = yaml.FullLoader )
        if not hasattr(cls, 'CONFIG_DATA'):
            setattr(cls, 'CONFIG_DATA', {})
        if isinstance(data, dict):
            # 将读取到的 dict 每个 key 都融合到 CONFIG_DATA 中
            getattr(cls, 'CONFIG_DATA').update(data)
        else:
            raise TypeError("仅支持字典类型的配置数据，请检查配置文件内容。")

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



