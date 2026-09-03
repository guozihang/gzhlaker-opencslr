"""VAC 损失的兼容导出。"""

from .criterions import CTCWithDistillationLoss


class VACLoss(CTCWithDistillationLoss):
    """VAC 模型的 CTC + 蒸馏损失。"""

    def __init__(self, loss_weights):
        super().__init__(loss_weights, who="VACLoss")
