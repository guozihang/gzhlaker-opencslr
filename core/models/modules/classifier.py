"""分类器模块，用于将序列特征映射到类别空间。

提供线性分类层，将时序建模后的特征转换为每个时间步的类别 logits。
"""

import torch.nn as nn
from .norm import NormLinear
from models.keys import Keys, require


class Classifier ( nn.Module ) :
    """线性分类器模块。

    将时序特征通过全连接层映射到类别空间，输出每个时间步的分类 logits。

    Args:
        args: 配置字典，需包含 "num_classes" 指定分类数。
    """

    def __init__ ( self , args ) :
        super ( Classifier , self ).__init__ ( )
        self.classifier = nn.Linear ( 1024 , args["num_classes"] )

    def forward ( self , data) :
        """前向传播。

        从数据字典中提取预测特征，通过线性层映射为分类 logits。

        Args:
            data: 数据字典，需包含 Keys.PREDICTIONS 特征张量。

        Returns:
            dict: 包含 Keys.SEQUENCE_LOGITS 的字典，logits 形状为 (T, B, num_classes)
        """
        require ( data , Keys.PREDICTIONS , who = "Classifier" )
        return {
            Keys.SEQUENCE_LOGITS: self.classifier ( data[Keys.PREDICTIONS] )
        }