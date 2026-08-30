"""OpenCSLR 设备管理器模块。

负责 GPU 设备的配置与管理，包括设备选择、CUDA 可见设备设置、
以及将数据张量迁移到指定设备的功能。
"""

import os

import torch


class DeviceManager(object):
    """设备管理器。

    管理 GPU 设备的初始化与数据迁移操作，支持多 GPU 配置，
    并提供将张量自动迁移到指定设备的工具方法。
    """
    @classmethod
    def init(cls, device):
        """初始化设备管理器，配置 GPU 设备。

        设置 CUDA 可见设备列表，并确定输出设备。
        支持单个 GPU 和多个 GPU 的配置。

        Args:
            device: 设备 ID 字符串，如 "0"、"0,1" 或 "None"
        """
        device = str ( device )
        if device != 'None' :
            cls.gpu_list = [ i for i in range ( len ( device.split ( ',' ) ) ) ]
            os.environ [ "CUDA_VISIBLE_DEVICES" ] = device
            output_device = cls.gpu_list [ 0 ]
        cls.output_device = output_device if len ( cls.gpu_list ) > 0 else "cpu"

    @classmethod
    def to(cls, data):
        """将数据张量迁移到指定设备。

        支持 FloatTensor、DoubleTensor、ByteTensor、LongTensor 以及
        列表/元组等数据类型的自动迁移。

        Args:
            data: 输入数据，支持张量或张量列表/元组

        Returns:
            迁移到指定设备后的数据

        Raises:
            ValueError: 当数据类型不支持时抛出
        """
        if isinstance(data, torch.FloatTensor):
            return data.to(cls.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(cls.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(cls.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(cls.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [cls.to(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))