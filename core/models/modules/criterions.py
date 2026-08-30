"""Loss functions for CSLR models.

Provides sequence-level loss functions including knowledge distillation
loss with label smoothing (SeqKD).
"""

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


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

