# -*- coding: utf-8 -*-
"""VAC (Visual Alignment Constraint) 模型。

组装:空间 ResNet → 时序 TemporalConv1D(VACTemporalConv) + BiLSTM + Classifier
      → VACLoss → Decoder。

VAC 的时序卷积与分类器共享权重:时序卷积的 fc 直接指向 classifier.classifier。
"""

import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import (BiLSTM, CTCLoss, Classifier, Decoder, NormLinear, ResNet,
                     SeqKD, TemporalConv1D, VACTemporalConv)


class VACLoss(nn.Module):
    """VAC 的损失:ConvCTC + SeqCTC + 序列蒸馏,按 loss_weights 加权。"""

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.ctc = CTCLoss()
        self.kd = SeqKD(T=8)

    def forward(self, data):
        require(data, Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL,
                Keys.FEAT_LEN, Keys.LABEL_LGT, who="VACLoss")
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


@register_model('vac')
def build_vac(args, gloss_dict, loss_weights):
    """构建 VAC (Visual Alignment Constraint) 模型。

    注意:VAC 的时序卷积与分类器共享权重(NormLinear),且 conv1d 的 fc
    直接指向 classifier.classifier 的引用。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 ResNet,时序模块为
        TemporalConv1D(VACTemporalConv) + BiLSTM + Classifier,损失为 VACLoss。
    """
    conv1d = TemporalConv1D(args, VACTemporalConv)
    classifier = Classifier(args)
    classifier.classifier = NormLinear(1024, args["num_classes"])
    conv1d.conv1d.fc = classifier.classifier
    return SignLanguageModel(
        spatial_module_container=Container([
            ResNet(args)
        ]),
        temporal_module_container=Container([
            conv1d,
            BiLSTM(args),
            classifier
        ]),
        loss_module_container=Container([
            VACLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
