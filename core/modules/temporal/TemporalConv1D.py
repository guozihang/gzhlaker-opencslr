"""TemporalConv1D 时序卷积模块包装器。

把帧级特征从数据字典取出、交给具体的时序卷积实现、再把结果写回字典。
具体用哪个卷积实现由调用方注入:TLP/SEN 用 ``tconv.TemporalConv``(带 LiftPool
与辅助损失),VAC 用 ``tconv.VACTemporalConv``(标准 MaxPool1d)。两者的差异在
各自的实现里,本包装器不做分支。
"""

import torch.nn as nn

from models import Keys, require

from .tconv import TemporalConv


class TemporalConv1D(nn.Module):
    """一维时序卷积模块。

    封装一个时序卷积实现,从数据字典中读取帧级特征进行时序建模。

    Args:
        args: 配置字典,需包含:
            kernel_size: 卷积核大小列表
            stride: 步长列表
            use_bn: 是否使用批归一化
            num_classes: 分类数(用于可选的分类头)
        conv_cls: 时序卷积实现类,需接受与 ``tconv.TemporalConv``
            相同的构造参数。
    """

    def __init__(self, args, conv_cls=TemporalConv):
        super(TemporalConv1D, self).__init__()
        self.conv1d = conv_cls(
            input_size=512,
            hidden_size=1024,
            kernel_size=args["kernel_size"],
            stride=args["stride"],
            use_bn=args["use_bn"],
            num_classes=args["num_classes"],
        )

    def forward(self, data):
        """前向传播。

        从数据字典中提取帧级特征和视频长度,通过时序卷积处理。

        Args:
            data: 数据字典,需包含:
                Keys.FRAMEWISE_FEATURES: 帧级特征,形状 (B, C, T)
                Keys.VID_LGT: 每个视频的实际帧数

        Returns:
            dict: 包含视觉特征、卷积 logits 和更新后特征长度的字典
        """
        require(data, Keys.FRAMEWISE_FEATURES, Keys.VID_LGT, who="TemporalConv1D")
        return self.conv1d(data[Keys.FRAMEWISE_FEATURES], data[Keys.VID_LGT])
