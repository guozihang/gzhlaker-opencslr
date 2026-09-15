# -*- coding: utf-8 -*-
"""CorrNet (Correlation Network) 模型。

组装:空间 corrnet_resnet18 → 时序 CorrNeT_TemporalConv1D + BiLSTM + Classifier
      → CorrNetLoss → Decoder。

支持可选的 weight_norm(分类头用 NormLinear)与 share_classifier
(共享时 conv1d 的 fc 与 classifier 的 classifier 是同一个模块)。
"""

import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import (BiLSTM, CTCLoss, Classifier, CorrNeT_TemporalConv1D,
                     Decoder, NormLinear, SeqKD, corrnet_resnet18)


class CorrNetLoss(nn.Module):
    """CorrNet 的损失:ConvCTC + SeqCTC + 序列蒸馏,按 loss_weights 加权。"""

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.ctc = CTCLoss()
        self.kd = SeqKD(T=8)

    def forward(self, data):
        require(data, Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL,
                Keys.FEAT_LEN, Keys.LABEL_LGT, who="CorrNetLoss")
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


@register_model('corrnet')
def build_corrnet(args, gloss_dict, loss_weights):
    """构建 CorrNet (Correlation Network) 模型。

    CorrNet 支持可选的权重归一化(weight_norm)和分类器共享(share_classifier),
    当 share_classifier=True 时 conv1d 的 fc 与 classifier 的 classifier 共享参数。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 corrnet_resnet18,时序模块为
        CorrNeT_TemporalConv1D + BiLSTM + Classifier,损失为 CorrNetLoss。
    """
    conv1d = CorrNeT_TemporalConv1D(args)
    classifier = Classifier(args)
    hidden_size = args.get("hidden_size", 1024)
    num_classes = args["num_classes"]

    if args.get("weight_norm", True):
        classifier.classifier = NormLinear(hidden_size, num_classes)
        conv1d.conv1d.fc = NormLinear(hidden_size, num_classes)
    else:
        classifier.classifier = nn.Linear(hidden_size, num_classes)
        conv1d.conv1d.fc = nn.Linear(hidden_size, num_classes)

    if args.get("share_classifier", True):
        conv1d.conv1d.fc = classifier.classifier

    return SignLanguageModel(
        spatial_module_container=Container([
            corrnet_resnet18(args)
        ]),
        temporal_module_container=Container([
            conv1d,
            BiLSTM(args),
            classifier
        ]),
        loss_module_container=Container([
            CorrNetLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
