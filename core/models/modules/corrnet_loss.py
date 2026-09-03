"""CorrNet 损失的兼容导出。"""

from .criterions import CTCWithDistillationLoss


class CorrNetLoss(CTCWithDistillationLoss):
    """CorrNet 模型的 CTC + 蒸馏损失。"""

    def __init__(self, loss_weights, blank_id=0):
        super().__init__(loss_weights, blank_id=blank_id, who="CorrNetLoss")
