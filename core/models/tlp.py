# -*- coding: utf-8 -*-
"""TLP (Temporal Lift Pooling) 模型。

组装:空间 ResNet → 时序 TemporalConv1D + BiLSTM + Classifier
      → TLPLoss → Decoder。
"""

import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import (BiLSTM, CTCLoss, Classifier, Decoder, ResNet, SeqKD,
                     TemporalConv1D)


class TLPLoss(nn.Module):
    """TLP 的损失:ConvCTC + SeqCTC + 序列蒸馏 + LiftPool 正则(Cu / Cp)。

    比 SEN/VAC/CorrNet 的损失多出 Cu、Cp 两项,它们由 LiftPool 模块算好后
    经 LOSS_LIFTPOOL_U / LOSS_LIFTPOOL_P 放进 data,这里只负责加权。
    """

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.ctc = CTCLoss()
        self.kd = SeqKD(T=8)

    def forward(self, data):
        """按 loss_weights 加权求和各项损失。

        Args:
            data: 数据字典,需含 CONV_LOGITS / SEQUENCE_LOGITS / LABEL /
                FEAT_LEN / LABEL_LGT;loss_weights 里出现 "Cu" / "Cp" 时
                还需对应的 LOSS_LIFTPOOL_U / LOSS_LIFTPOOL_P。

        Returns:
            dict: Keys.LOSS(总损失)与 Keys.TOTAL_LOSS(各分项)。
        """
        required = [Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL,
                    Keys.FEAT_LEN, Keys.LABEL_LGT]
        if "Cu" in self.loss_weights:
            required.append(Keys.LOSS_LIFTPOOL_U)
        if "Cp" in self.loss_weights:
            required.append(Keys.LOSS_LIFTPOOL_P)
        require(data, *required, who="TLPLoss")
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
            elif key == "Cu":
                total_loss["Cu"] = weight * data[Keys.LOSS_LIFTPOOL_U]
                loss += total_loss["Cu"]
            elif key == "Cp":
                total_loss["Cp"] = weight * data[Keys.LOSS_LIFTPOOL_P]
                loss += total_loss["Cp"]
        return {Keys.LOSS: loss, Keys.TOTAL_LOSS: total_loss}


@register_model('tlp')
def build_tlp(args, gloss_dict, loss_weights):
    """构建 TLP (Temporal Lift Pooling) 模型。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 ResNet,时序模块为
        TemporalConv1D + BiLSTM + Classifier,损失为 TLPLoss。
    """
    return SignLanguageModel(
        spatial_module_container=Container([
            ResNet(args)
        ]),
        temporal_module_container=Container([
            TemporalConv1D(args),
            BiLSTM(args),
            Classifier(args)
        ]),
        loss_module_container=Container([
            TLPLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
