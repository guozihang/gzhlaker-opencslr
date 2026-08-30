"""VAC 时序卷积模块包装器。

基于 tconv.VACTemporalConv 构建的时序卷积模块，
用于 VAC 模型中的帧级特征时序建模。
"""

import torch.nn as nn
from .tconv import VACTemporalConv
from models.keys import Keys, require


class VACTemporalConv1D ( nn.Module ) :
    """VAC 模型的一维时序卷积模块。

    封装 VACTemporalConv，配置输入大小、隐藏大小、卷积核等参数，
    从数据字典中读取帧级特征进行时序建模。

    Args:
        args: 配置字典，需包含:
            kernel_size: 卷积核大小列表
            stride: 步长列表
            use_bn: 是否使用批归一化
            num_classes: 分类数（用于可选的分类头）
    """

    def __init__ ( self , args ) :
        super ( VACTemporalConv1D , self ).__init__ ( )
        self.conv1d = VACTemporalConv (
            input_size = 512 ,
            hidden_size = 1024 ,
            kernel_size = args["kernel_size"] ,
            stride = args["stride"] ,
            use_bn = args["use_bn"] ,
            num_classes = args["num_classes"] ,
        )

    def forward ( self , data) :
        """前向传播。

        从数据字典中提取帧级特征和视频长度，通过时序卷积处理。

        Args:
            data: 数据字典，需包含:
                Keys.FRAMEWISE_FEATURES: 帧级特征，形状 (B, C, T)
                Keys.VID_LGT: 每个视频的实际帧数

        Returns:
            dict: 包含视觉特征、卷积 logits 和特征长度的字典
        """
        require ( data , Keys.FRAMEWISE_FEATURES , Keys.VID_LGT , who = "VACTemporalConv1D" )
        return self.conv1d ( data[Keys.FRAMEWISE_FEATURES] , data[Keys.VID_LGT] )