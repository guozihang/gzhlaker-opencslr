import torch
import torch.nn as nn
from .criterions import SeqKD
from models.keys import Keys, require


class CorrNetLoss(nn.Module):
    def __init__(self, loss_weights, blank_id=0):
        super().__init__()
        self.loss_weights = loss_weights
        self.loss = {}
        self.loss["CTCLoss"] = torch.nn.CTCLoss(
            blank=blank_id,
            reduction="none",
            zero_infinity=False,
        )
        self.loss["distillation"] = SeqKD(T=8)

    def forward(self, data):
        require(data, Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL, Keys.FEAT_LEN, Keys.LABEL_LGT,
                who="CorrNetLoss")
        loss = 0
        total_loss = {}

        for k, weight in self.loss_weights.items():
            if k == "ConvCTC":
                total_loss["ConvCTC"] = weight * self.loss["CTCLoss"](
                    data[Keys.CONV_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int(),
                ).mean()
                loss += total_loss["ConvCTC"]
            elif k == "SeqCTC":
                total_loss["SeqCTC"] = weight * self.loss["CTCLoss"](
                    data[Keys.SEQUENCE_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int(),
                ).mean()
                loss += total_loss["SeqCTC"]
            elif k == "Dist":
                total_loss["Dist"] = weight * self.loss["distillation"](
                    data[Keys.CONV_LOGITS],
                    data[Keys.SEQUENCE_LOGITS].detach(),
                    use_blank=False,
                )
                loss += total_loss["Dist"]

        return {
            Keys.LOSS: loss,
            Keys.TOTAL_LOSS: total_loss,
        }
