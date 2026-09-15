# -*- coding: utf-8 -*-
"""SEN (Stochastic Encoding Network) 模型。

组装:空间 SENresnet → 时序 TemporalConv1D + BiLSTM + Classifier
      → SENLoss → Decoder。
与 TLP 的差别只在空间网络和损失。
"""

import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import (BiLSTM, CTCLoss, Classifier, Decoder, SENresnet, SeqKD,
                     TemporalConv1D)


class SENLoss(nn.Module):
    """SEN 的损失:ConvCTC + SeqCTC + 序列蒸馏,按 loss_weights 加权。"""

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.ctc = CTCLoss()
        self.kd = SeqKD(T=8)

    def forward(self, data):
        require(data, Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL,
                Keys.FEAT_LEN, Keys.LABEL_LGT, who="SENLoss")
        loss = 0
        total_loss = {}
        for key, weight in self.loss_weights.items():
            if key == "ConvCTC":
                total_loss["ConvCTC"] = weight * self.ctc(data[Keys.CONV_LOGITS], data)
                loss += total_loss["ConvCTC"]
            elif key == "SeqCTC":
                total_loss["SeqCTC"] = weight * self.ctc(data[Keys.SEQUENCE_LOGITS], data)
                loss += total_loss["SeqCTC"]
            elif key == "Dist":
                total_loss["Dist"] = weight * self.kd(
                    data[Keys.CONV_LOGITS], data[Keys.SEQUENCE_LOGITS].detach(),
                    use_blank=False)
                loss += total_loss["Dist"]
        return {Keys.LOSS: loss, Keys.TOTAL_LOSS: total_loss}


@register_model('sen')
def build_sen(args, gloss_dict, loss_weights):
    """构建 SEN (Stochastic Encoding Network) 模型。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 SENresnet,时序模块为
        TemporalConv1D + BiLSTM + Classifier,损失为 SENLoss。
    """
    return SignLanguageModel(
        spatial_module_container=Container([
            SENresnet(args)
        ]),
        temporal_module_container=Container([
            TemporalConv1D(args),
            BiLSTM(args),
            Classifier(args)
        ]),
        loss_module_container=Container([
            SENLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
