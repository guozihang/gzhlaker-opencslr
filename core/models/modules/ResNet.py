"""ResNet 骨干网络，用于空间特征提取。

基于 torchvision 预训练的 ResNet 模型，用于从视频帧中提取帧级特征。
"""

import torch
import torch.nn as nn
import torchvision.models as models
from .identity import Identity
from models.keys import Keys, require


class ResNet( nn.Module ) :
    """基于 ResNet 的空间特征提取模块。

    使用预训练的 ResNet（如 ResNet18）对视频帧进行空间特征提取。
    支持视频输入（5维张量）和帧级特征输入（已提取的特征向量）。
    替换全连接层为 Identity 以保留特征向量。

    Args:
        args: 配置字典，需包含 "c2d_type" 指定 ResNet 类型（如 "resnet18"）。
    """

    def __init__ ( self, args) :
        super ( ResNet , self ).__init__ ( )
        c2d_type = args["c2d_type"]
        self.conv2d = getattr ( models , c2d_type ) ( weights = "ResNet18_Weights.IMAGENET1K_V1" )
        self.conv2d.fc = Identity ()

    def masked_bn ( self , inputs , len_x ) :
        """对变长序列进行带掩码的批量归一化处理。

        根据实际长度对输入进行拼接，通过 ResNet 前向传播后，
        再按原始长度填充回统一长度，避免 padding 区域影响 BN 统计量。

        Args:
            inputs: 输入张量，形状为 (batch * temp, channel, height, width)
            len_x: 每个样本的实际帧数列表，长度为 batch

        Returns:
            经过 ResNet 处理后的特征张量
        """

        def pad ( tensor , length ) :
            return torch.cat (
                [ tensor , tensor.new ( length - tensor.size ( 0 ) , *tensor.size ( ) [ 1 : ] ).zero_ ( ) ] )

        x = torch.cat ( [ inputs [ len_x [ 0 ] * idx :len_x [ 0 ] * idx + lgt ] for idx , lgt in enumerate ( len_x ) ] )
        x = self.conv2d ( x )
        x = torch.cat ( [ pad ( x [ sum ( len_x [ :idx ] ) :sum ( len_x [ :idx + 1 ] ) ] , len_x [ 0 ] )
                          for idx , lgt in enumerate ( len_x ) ] )
        return x

    def forward(self , data) :
        """前向传播。

        从数据字典中获取视频帧和长度信息，通过 ResNet 提取帧级特征。

        Args:
            data: 数据字典，需包含:
                Keys.VID: 视频帧张量，形状为 (B, T, C, H, W) 或 (B*T, C, H, W)
                Keys.VID_LGT: 每个视频的实际帧数

        Returns:
            dict: 包含 Keys.FRAMEWISE_FEATURES 的字典，特征形状为 (B, C, T)
        """
        require ( data , Keys.VID , Keys.VID_LGT , who = "ResNet" )
        x = data[Keys.VID]
        len_x = data[Keys.VID_LGT]
        if len ( x.shape ) == 5 :
            # videos
            batch , temp , channel , height , width = x.shape
            inputs = x.reshape ( batch * temp , channel , height , width )
            framewise = self.masked_bn ( inputs , len_x )
            framewise = framewise.reshape ( batch , temp , -1 ).transpose ( 1 , 2 )
        else :
            # frame-wise features
            framewise = x
        return {
            Keys.FRAMEWISE_FEATURES: framewise,
        }