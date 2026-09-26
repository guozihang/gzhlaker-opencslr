# -*- coding: utf-8 -*-
"""SEN (Stochastic Encoding Network) 模型。

组装:空间 SENresnet → 时序 TemporalConv1D + BiLSTM + Classifier
      → SENLoss → Decoder。
与 TLP 的差别只在空间网络和损失。

时序卷积有两种实现,由 ``model_args.temporal_conv`` 选择:

    maxpool         重构前(ebdb77f)的 SEN 用的 MaxPool 版,结构等同
                    senmodules/sen_TemporalConv.py,也是唯一有历史权重的版本。
    liftpool(默认)  当前实现:LiftPool 版 TemporalConv。

两者参数集合完全不同,旧 SEN checkpoint 必须用 ``maxpool`` 才加载得上。
"""

import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import (BiLSTM, CTCLoss, Classifier, Decoder, SENresnet, SeqKD,
                     TemporalConv1D, VACTemporalConv)

# 旧 sen_TemporalConv 的 conv_type → kernel_size 映射,原样保留。
_LEGACY_CONV_TYPES = {
    0: ['K3'],
    1: ['K5', 'P2'],
    2: ['K5', 'P2', 'K5', 'P2'],
}


def _temporal_args(args, conv_type_wins):
    """补齐时序卷积的构造参数,两种实现共用。

    ``use_bn`` / ``num_classes`` 直接来自配置;``stride`` 旧 SEN 不需要,当前
    TemporalConv1D 要,所以补个默认;``kernel_size`` 与 ``conv_type`` 的优先级
    在两种实现里是相反的——旧 SEN 只在配置里**没有** conv_type 时才用
    kernel_size(``conv_type_wins=True``),当前版本反过来,没给 kernel_size 才按
    conv_type 推。两边都保留,旧 checkpoint 才不会建错结构。
    """
    conv_args = dict(args)
    conv_args.setdefault("stride", [0])
    if "conv_type" in args and (conv_type_wins or not conv_args.get("kernel_size")):
        conv_type = args["conv_type"]
        if conv_type not in _LEGACY_CONV_TYPES:
            raise ValueError("SEN 的 conv_type 只支持 {},收到 {!r}".format(
                sorted(_LEGACY_CONV_TYPES), conv_type))
        conv_args["kernel_size"] = list(_LEGACY_CONV_TYPES[conv_type])
    elif not conv_args.get("kernel_size"):
        conv_args["kernel_size"] = list(_LEGACY_CONV_TYPES[2])
    return conv_args


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
        args: 配置字典,包含模型超参数。``temporal_conv`` 为 "maxpool" 时
            时序模块按重构前的 MaxPool 版构建(用于加载旧 checkpoint),
            否则用 LiftPool 版。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 SENresnet,时序模块为
        TemporalConv1D + BiLSTM + Classifier,损失为 SENLoss。
    """
    if args.get("temporal_conv", "liftpool") == "maxpool":
        # 旧 sen_TemporalConv 的等价实现就是 VACTemporalConv,不再复制一份;
        # 它把 kernel_size 各层摊平成一个 nn.Sequential,key 与旧代码逐字相同。
        time_conv = TemporalConv1D(
            _temporal_args(args, conv_type_wins=True), VACTemporalConv,
            input_size=args.get("input_size", 512),
            hidden_size=args.get("hidden_size", 1024))
    else:
        time_conv = TemporalConv1D(_temporal_args(args, conv_type_wins=False))

    return SignLanguageModel(
        spatial_module_container=Container([
            SENresnet(args)
        ]),
        temporal_module_container=Container([
            time_conv,
            BiLSTM(args),
            Classifier(args)
        ]),
        loss_module_container=Container([
            SENLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
