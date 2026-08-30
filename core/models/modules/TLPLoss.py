"""TLP (Temporal LiftPool) 损失函数模块。

支持 CTC 损失、知识蒸馏损失以及 LiftPool 正则化损失（Cu 和 Cp）的加权组合。
"""

import torch
import torch.nn as nn
from .criterions import SeqKD
from models.keys import Keys, require


class TLPLoss( nn.Module ) :
    """TLP 模型的损失函数模块。

    根据损失权重字典，组合计算 CTC 损失（ConvCTC、SeqCTC）、
    知识蒸馏损失（Dist）和 LiftPool 正则化损失（Cu、Cp）。

    Args:
        loss_weights: 字典，键为损失名称，值为对应权重。
            支持的键: "ConvCTC", "SeqCTC", "Dist", "Cu", "Cp"
    """

    def __init__ ( self, loss_weights) :
        super ( TLPLoss , self ).__init__ ( )
        self.loss = {}
        self.loss_weights = loss_weights
        self.loss [ 'CTCLoss' ] = torch.nn.CTCLoss ( reduction = 'none' , zero_infinity = False )
        self.loss [ 'distillation' ] = SeqKD ( T = 8 )

    def forward ( self , data ) :
        """前向传播计算损失。

        根据损失权重配置，从数据字典中读取相应数据并计算各项损失。

        Args:
            data: 数据字典，需包含:
                Keys.CONV_LOGITS: 卷积层输出 logits
                Keys.SEQUENCE_LOGITS: 序列模型输出 logits
                Keys.LABEL: 真实标签
                Keys.FEAT_LEN: 特征长度
                Keys.LABEL_LGT: 标签长度
                如果 loss_weights 包含 "Cu"，还需 Keys.LOSS_LIFTPOOL_U
                如果 loss_weights 包含 "Cp"，还需 Keys.LOSS_LIFTPOOL_P

        Returns:
            dict: 包含 Keys.LOSS（总损失）和 Keys.TOTAL_LOSS（各分项损失字典）
        """
        required = [ Keys.CONV_LOGITS , Keys.SEQUENCE_LOGITS , Keys.LABEL , Keys.FEAT_LEN , Keys.LABEL_LGT ]
        if 'Cu' in self.loss_weights :
            required.append ( Keys.LOSS_LIFTPOOL_U )
        if 'Cp' in self.loss_weights :
            required.append ( Keys.LOSS_LIFTPOOL_P )
        require ( data , *required , who = "TLPLoss" )
        loss = 0
        total_loss = {}
        for k , weight in self.loss_weights.items ( ) :
            if k == 'ConvCTC' :
                total_loss [ 'ConvCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    data [ Keys.CONV_LOGITS ].log_softmax ( -1 ) ,
                    data[Keys.LABEL].cpu ( ).int ( ) , data [ Keys.FEAT_LEN ].cpu ( ).int ( ) ,
                    data[Keys.LABEL_LGT].cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'ConvCTC' ]
            elif k == 'SeqCTC' :
                total_loss [ 'SeqCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    data [ Keys.SEQUENCE_LOGITS ].log_softmax ( -1 ) ,
                    data[Keys.LABEL].cpu ( ).int ( ) , data [ Keys.FEAT_LEN ].cpu ( ).int ( ) ,
                    data[Keys.LABEL_LGT].cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'SeqCTC' ]
            elif k == 'Dist' :
                total_loss [ 'Dist' ] = weight * self.loss [ 'distillation' ] ( data [ Keys.CONV_LOGITS ] ,
                                                                                data [
                                                                                    Keys.SEQUENCE_LOGITS ].detach ( ) ,
                                                                                use_blank = False )
                loss += total_loss [ 'Dist' ]
            elif k == 'Cu' :
                total_loss [ 'Cu' ] = weight * data [ Keys.LOSS_LIFTPOOL_U ]
                loss += total_loss [ 'Cu' ]
            elif k == 'Cp' :
                total_loss [ 'Cp' ] = weight * data [ Keys.LOSS_LIFTPOOL_P ]
                loss += total_loss [ 'Cp' ]
        return {
            Keys.LOSS: loss,
            Keys.TOTAL_LOSS: total_loss
        }

