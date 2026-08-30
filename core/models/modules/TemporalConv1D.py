"""TemporalConv1D 时序卷积模块。

基于 TemporalConv 构建的一维时序卷积模块，用于对帧级特征进行时序建模。
"""

import torch.nn as nn
from .tconv import TemporalConv
from models.keys import Keys, require


class TemporalConv1D ( nn.Module ) :
    """一维时序卷积模块。

    封装 TemporalConv，配置卷积核大小、步长、批归一化和分类数等参数，
    从数据字典中读取帧级特征并输出时序建模后的视觉特征。

    Args:
        args: 配置字典，需包含:
            kernel_size: 卷积核大小列表
            stride: 步长列表
            use_bn: 是否使用批归一化
            num_classes: 分类数（用于可选的分类头）
    """

    def __init__ ( self , args ) :
        super ( TemporalConv1D , self ).__init__ ( )
        self.conv1d = TemporalConv (
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
            dict: 包含视觉特征、卷积 logits 和更新后特征长度的字典
        """
        require ( data , Keys.FRAMEWISE_FEATURES , Keys.VID_LGT , who = "TemporalConv1D" )
        return self.conv1d ( data[Keys.FRAMEWISE_FEATURES] , data[Keys.VID_LGT] )