# -*- coding: utf-8 -*-
"""SEN (Stochastic Encoding Network) 模型。

组装:空间 SENresnet → 时序 sen_TemporalConv(MaxPool 版) + BiLSTM + Classifier
      → SENLoss → sen_Decoder。
与 TLP 的差别在空间网络、时序卷积与损失。

时序卷积由 ``model_args.temporal_conv`` 选择:

    maxpool(默认)  与上游 ``senmodules/sen_TemporalConv.py`` 等价,也是重构前
                   本仓库 SEN 的结构 —— 上游与本仓库的历史 SEN 权重都是这个。
    liftpool       重构期间的本地变体,复用 TLP 的 LiftPool 版 ``TemporalConv``,
                   只有用该结构训出来的权重才需要它。

两种结构的参数集合完全不同,选错会直接加载不上权重,所以默认按上游来。
"""

import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import (BiLSTM, CTCLoss, Classifier, Decoder, SENresnet, SeqKD,
                     TemporalConv1D, sen_TemporalConv)


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


class SENDecoder(nn.Module):
    """SEN 的解码器(与上游 ``sen_Decoder`` 等价)。

    与通用 ``Decoder`` 的唯一区别是**训练期不解码**:上游 SEN 只在
    ``not self.training`` 时跑束搜索,训练时 ``recognized_sents`` 为 None。
    这不影响损失与梯度(训练循环只读 loss/total_loss),但省掉了每个 batch 的
    一次束搜索 —— 仓库内的束搜索是纯 Python 实现,这一步很贵。
    """

    def __init__(self, args, gloss_dict):
        super(SENDecoder, self).__init__()
        self.decoder = Decoder(args, gloss_dict)

    def forward(self, data):
        pred = None
        if not self.training:
            require(data, Keys.SEQUENCE_LOGITS, Keys.FEAT_LEN, who="SENDecoder")
            pred = self.decoder.decode(data[Keys.SEQUENCE_LOGITS], data[Keys.FEAT_LEN],
                                       batch_first=False, probs=False)
        return {
            Keys.RECOGNIZED_SENTS: pred
        }


def _build_temporal(args):
    """SEN 的时序卷积:默认对齐上游(MaxPool 版),``liftpool`` 为本地变体。"""
    mode = args.get("temporal_conv", "maxpool")
    if mode == "maxpool":
        return sen_TemporalConv(args)
    if mode == "liftpool":
        # 本地变体:复用 TLP 的 LiftPool 版包装器,需要配置里给 kernel_size/stride
        return TemporalConv1D(args)
    raise ValueError(
        "SEN 的 temporal_conv 只支持 'maxpool'(上游结构)/ 'liftpool'(本地变体),"
        "收到 {!r}".format(mode)
    )


@register_model('sen')
def build_sen(args, gloss_dict, loss_weights):
    """构建 SEN (Stochastic Encoding Network) 模型。

    Args:
        args: 配置字典,包含模型超参数;``temporal_conv`` 选时序卷积结构
            (默认 "maxpool",与上游一致)。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 SENresnet,时序模块为
        sen_TemporalConv + BiLSTM + Classifier,损失为 SENLoss。
    """
    return SignLanguageModel(
        spatial_module_container=Container([
            SENresnet(args)
        ]),
        temporal_module_container=Container([
            _build_temporal(args),
            BiLSTM(args),
            Classifier(args)
        ]),
        loss_module_container=Container([
            SENLoss(loss_weights)
        ]),
        decoder=SENDecoder(args, gloss_dict)
    )
