"""SEN 模型的损失函数模块。

计算 CTC 损失（用于卷积层和序列层输出）以及知识蒸馏损失，
支持多任务学习中的加权组合。
"""
import torch
from torch import nn
from .criterions import SeqKD
from ..keys import Keys, require


class SENLoss(nn.Module):
    """SEN 损失函数模块。

    计算 CTC 损失（ConvCTC 和 SeqCTC）以及知识蒸馏损失（Dist），
    并根据给定的权重对各项损失进行加权求和。

    Args:
        loss_weights: 字典，包含各损失项的权重，如 {'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 0.5}
    """

    def __init__(self,loss_weights):
        super(SENLoss,self).__init__()
        self.loss={}
        self.loss_weights=loss_weights
        self.loss['CTCLoss']=nn.CTCLoss(reduction='none',zero_infinity=False)
        self.loss['Distillation']=SeqKD(T=8)
    def forward(self,data):
        """前向传播（计算损失）。

        Args:
            data: 包含 CONV_LOGITS, SEQUENCE_LOGITS, LABEL, FEAT_LEN, LABEL_LGT 的字典

        Returns:
            dict: 包含总损失和各项损失明细的字典
        """
        require(data, Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL, Keys.FEAT_LEN, Keys.LABEL_LGT,
                who="SENLoss")
        loss=0
        total_loss={}
        for k,weight in self.loss_weights.items():
            if k=='ConvCTC':
                total_loss['ConvCTC']=weight*self.loss['CTCLoss'](
                    data[Keys.CONV_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int()
                ).mean()
                loss+=total_loss['ConvCTC']
            elif k=='SeqCTC':
                total_loss['SeqCTC']=weight*self.loss['CTCLoss'](
                    data[Keys.SEQUENCE_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int()
                ).mean()
                loss+=total_loss['SeqCTC']
            elif k=='Dist':
                total_loss['Dist']=weight*self.loss['Distillation'](
                    data[Keys.CONV_LOGITS],data[Keys.SEQUENCE_LOGITS].detach(),use_blank=False
                )
                loss+=total_loss['Dist']
        return {
            Keys.LOSS:loss,
            Keys.TOTAL_LOSS:total_loss
        }
