"""Loss functions for CSLR models.

Provides sequence-level loss functions including knowledge distillation
loss with label smoothing (SeqKD).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.keys import Keys, require


class SeqKD(nn.Module):
    """
    NLL loss with label smoothing via knowledge distillation.

    Computes KL divergence between the student's log-probabilities and
    the teacher's soft probabilities, scaled by temperature T.

    Args:
        T: Temperature for softening the probability distributions (default 1).
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        """Compute distillation loss.

        Args:
            prediction_logits: Student network logits, shape (B, T, N).
            ref_logits: Teacher network logits, shape (B, T, N).
            use_blank: If True, include blank class in loss computation.

        Returns:
            torch.Tensor: Scalar distillation loss.
        """
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        # mask_probs = F.softmax(ref_logits[:, :, 1:], dim=-1).view(-1, ref_logits.shape[2] - 1)
        # mask = torch.max(mask_probs, dim=1)[0] > 0.5
        # if torch.sum(mask) != 0:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask) / torch.sum(mask)
        # else:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask)
        return loss


class CTCWithDistillationLoss(nn.Module):
    """公共的 CTC + 序列蒸馏损失。

    SEN、VAC 和 CorrNet 三个模型的损失结构完全一致，统一实现于此。
    """

    def __init__(self, loss_weights, blank_id=0, who="CTCWithDistillationLoss"):
        super().__init__()
        self.loss_weights = loss_weights
        self.who = who
        self.loss = {
            "CTCLoss": torch.nn.CTCLoss(
                blank=blank_id, reduction="none", zero_infinity=False
            ),
            "distillation": SeqKD(T=8),
        }

    def forward(self, data):
        require(
            data,
            Keys.CONV_LOGITS,
            Keys.SEQUENCE_LOGITS,
            Keys.LABEL,
            Keys.FEAT_LEN,
            Keys.LABEL_LGT,
            who=self.who,
        )
        loss = 0
        total_loss = {}

        for key, weight in self.loss_weights.items():
            if key == "ConvCTC":
                total_loss["ConvCTC"] = weight * self.loss["CTCLoss"](
                    data[Keys.CONV_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int(),
                ).mean()
                loss += total_loss["ConvCTC"]
            elif key == "SeqCTC":
                total_loss["SeqCTC"] = weight * self.loss["CTCLoss"](
                    data[Keys.SEQUENCE_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int(),
                ).mean()
                loss += total_loss["SeqCTC"]
            elif key == "Dist":
                total_loss["Dist"] = weight * self.loss["distillation"](
                    data[Keys.CONV_LOGITS],
                    data[Keys.SEQUENCE_LOGITS].detach(),
                    use_blank=False,
                )
                loss += total_loss["Dist"]

        return {Keys.LOSS: loss, Keys.TOTAL_LOSS: total_loss}
