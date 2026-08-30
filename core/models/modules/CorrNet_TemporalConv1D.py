"""CorrNet 时序卷积模块包装器。

基于 corrnet_tconv.TemporalConv 构建的时序卷积模块，
用于 CorrNet 模型中的帧级特征时序建模。
"""

import torch.nn as nn
from .corrnet_tconv import TemporalConv
from models.keys import Keys, require


class CorrNeT_TemporalConv1D(nn.Module):
    """CorrNet 的一维时序卷积模块。

    封装 corrnet_tconv.TemporalConv，配置输入大小、隐藏大小、
    卷积类型等参数，从数据字典中读取帧级特征进行时序建模。

    Args:
        args: 配置字典，可包含:
            input_size: 输入特征维度（默认 512）
            hidden_size: 隐藏层维度（默认 1024）
            conv_type: 卷积类型（默认 2）
            use_bn: 是否使用批归一化（默认 False）
            num_classes: 分类数（默认 -1，不添加分类头）
    """

    def __init__(self, args):
        super(CorrNeT_TemporalConv1D, self).__init__()
        self.conv1d = TemporalConv(
            input_size=args.get("input_size", 512),
            hidden_size=args.get("hidden_size", 1024),
            conv_type=args.get("conv_type", 2),
            use_bn=args.get("use_bn", False),
            num_classes=args.get("num_classes", -1),
        )

    def forward(self, data):
        """前向传播。

        从数据字典中提取帧级特征和视频长度，通过时序卷积处理。

        Args:
            data: 数据字典，需包含:
                Keys.FRAMEWISE_FEATURES: 帧级特征，形状 (B, C, T)
                Keys.VID_LGT: 每个视频的实际帧数

        Returns:
            dict: 包含视觉特征、卷积 logits 和特征长度的字典
        """
        require(data, Keys.FRAMEWISE_FEATURES, Keys.VID_LGT, who="CorrNeT_TemporalConv1D")
        output = self.conv1d(data[Keys.FRAMEWISE_FEATURES], data[Keys.VID_LGT])
        return output