"""BiLSTM 时序建模模块包装器。

基于 BiLSTMLayer 构建的双层 BiLSTM 时序模型，用于处理视觉特征序列。
"""

import torch.nn as nn
from .BiLSTM import BiLSTMLayer
from models.keys import Keys, require


class BiLSTM( nn.Module ) :
    """BiLSTM 时序建模模块。

    封装 BiLSTMLayer，配置为 2 层双向 LSTM，隐藏层大小为 1024，
    从数据字典中读取视觉特征并输出预测结果。

    Args:
        args: 配置字典（当前未使用，预留接口）。
    """

    def __init__ ( self , args) :
        super ( BiLSTM , self ).__init__ ( )
        self.temporal_model = BiLSTMLayer (
            rnn_type = 'LSTM' ,
            input_size = 1024 ,
            hidden_size = 1024 ,
            num_layers = 2 ,
            bidirectional = True
        )

    def forward ( self , data) :
        """前向传播。

        从数据字典中提取视觉特征和序列长度，通过 BiLSTM 建模时序依赖。

        Args:
            data: 数据字典，需包含:
                Keys.VISUAL_FEAT: 视觉特征，形状 (T, B, C)
                Keys.FEAT_LEN: 每个序列的实际长度，形状 (B,)

        Returns:
            dict: 包含 Keys.PREDICTIONS 和 Keys.HIDDEN 的字典
        """
        require ( data , Keys.VISUAL_FEAT , Keys.FEAT_LEN , who = "BiLSTM" )
        return self.temporal_model ( data [ Keys.VISUAL_FEAT ] , data [ Keys.FEAT_LEN ] )