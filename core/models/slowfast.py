# -*- coding: utf-8 -*-
"""SlowFast 模型。

组装:空间 SlowFast(慢+快双路径) → 时序 TemporalSlowFastConv1D + temporal_model
      → SlowFastLoss → SlowFastDecoder。

SlowFast 的时序部分是双通路的:conv1d 的 forward 返回 list,并由 temporal_model
共享同一组权重生成主通路 logits,所以 conv1d 要作为参数传给 temporal_model。
"""

import torch.nn as nn

from models import Container, Keys, SignLanguageModel, register_model, require
from modules import (CTCLoss, Decoder, SeqKD, SlowFast, TemporalSlowFastConv1D,
                     temporal_model)


class SlowFastLoss(nn.Module):
    """SlowFast 的损失:多通路 CTC + 序列蒸馏。

    与单通路模型不同,``SEQUENCE_LOGITS`` / ``CONV_LOGITS`` 是 list:
    [0] 主路径、[1] 慢路径、[2] 快路径。SeqCTC / ConvCTC / Dist 作用在
    主路径上;Slow / Fast 除了自己那一项,还会各带出 ConvCTC_{Slow,Fast}
    和 Dist_{Slow,Fast} 两项,权重分别取自配置里的 ConvCTC / Dist。
    """

    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        self.ctc = CTCLoss()
        self.kd = SeqKD(T=8)

    def forward(self, data):
        require(data, Keys.SEQUENCE_LOGITS, Keys.CONV_LOGITS, Keys.LABEL,
                Keys.FEAT_LEN, Keys.LABEL_LGT, who="SlowFastLoss")
        loss = 0
        total_loss = {}
        for key, weight in self.loss_weights.items():
            if key == "SeqCTC":
                total_loss["SeqCTC"] = weight * self.ctc(data[Keys.SEQUENCE_LOGITS][0], data)
                loss += total_loss["SeqCTC"]
            elif key == "Slow" or key == "Fast":
                i = 1 if key == "Slow" else 2
                if len(data[Keys.SEQUENCE_LOGITS]) <= i or len(data[Keys.CONV_LOGITS]) <= i:
                    continue
                total_loss[key] = weight * self.loss_weights.get("SeqCTC", 1.0) * self.ctc(
                    data[Keys.SEQUENCE_LOGITS][i], data)
                loss += total_loss[key]
                if "ConvCTC" in self.loss_weights:
                    total_loss[f"ConvCTC_{key}"] = weight * self.loss_weights["ConvCTC"] * self.ctc(
                        data[Keys.CONV_LOGITS][i], data)
                    loss += total_loss[f"ConvCTC_{key}"]
                if "Dist" in self.loss_weights:
                    total_loss[f"Dist_{key}"] = weight * self.loss_weights["Dist"] * self.kd(
                        data[Keys.CONV_LOGITS][i], data[Keys.SEQUENCE_LOGITS][i].detach(),
                        use_blank=False)
                    loss += total_loss[f"Dist_{key}"]
            elif key == "ConvCTC":
                total_loss["ConvCTC"] = weight * self.ctc(data[Keys.CONV_LOGITS][0], data)
                loss += total_loss["ConvCTC"]
            elif key == "Dist":
                total_loss["Dist"] = weight * self.kd(
                    data[Keys.CONV_LOGITS][0], data[Keys.SEQUENCE_LOGITS][0].detach(),
                    use_blank=False)
                loss += total_loss["Dist"]
        return {Keys.LOSS: loss, Keys.TOTAL_LOSS: total_loss}


class SlowFastDecoder(Decoder):
    """SlowFast 的解码器:只取主通路 logits 来解码。

    SlowFast 下 ``SEQUENCE_LOGITS`` 是双通路的 list(SlowFastLoss 需要整份),
    而解码只用主通路,所以这里取 ``[0]``。其余逻辑与 ``Decoder`` 完全相同。
    """

    def __call__(self, data):
        require(data, Keys.SEQUENCE_LOGITS, Keys.FEAT_LEN, who="SlowFastDecoder")
        pred = self.decode(data[Keys.SEQUENCE_LOGITS][0], data[Keys.FEAT_LEN],
                           batch_first=False, probs=False)
        return {
            Keys.RECOGNIZED_SENTS: pred
        }


@register_model('slowfast')
def build_slowfast(args, gloss_dict, loss_weights):
    """构建 SlowFast 模型。

    SlowFast 使用双路径空间网络(慢路径+快路径)和专用的时序卷积与解码器。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 SlowFast,时序模块为
        TemporalSlowFastConv1D + temporal_model,损失为 SlowFastLoss。
    """
    conv1d = TemporalSlowFastConv1D(args)

    return SignLanguageModel(
        spatial_module_container=Container([
            SlowFast(args)
        ]),

        temporal_module_container=Container([
            conv1d,
            temporal_model(args, conv1d),
        ]),

        loss_module_container=Container([
            SlowFastLoss(loss_weights)
        ]),

        decoder=SlowFastDecoder(args, gloss_dict)
    )
