"""CTC 损失单元。"""

import torch
import torch.nn as nn

from models import Keys


class CTCLoss(nn.Module):
    """CTC 损失单元:log_softmax → CTCLoss → 取 batch 均值。

    各模型原本都手写这四行(log_softmax、三次 .cpu().int()、.mean()),
    抽成单元后模型侧只需 ``ctc(logits, data)``。

    Args:
        blank: CTC blank 的下标(默认 0)。
    """

    def __init__(self, blank=0):
        super().__init__()
        self.ctc = torch.nn.CTCLoss(blank=blank, reduction="none", zero_infinity=False)

    def forward(self, logits, data):
        """计算 logits 相对标签的 CTC 损失均值。

        Args:
            logits: 网络输出 logits,形状 (T, B, N)。
            data: 数据字典,需含 LABEL / FEAT_LEN / LABEL_LGT。

        Returns:
            torch.Tensor: 标量 CTC 损失。
        """
        return self.ctc(
            logits.log_softmax(-1),
            data[Keys.LABEL].cpu().int(),
            data[Keys.FEAT_LEN].cpu().int(),
            data[Keys.LABEL_LGT].cpu().int(),
        ).mean()
