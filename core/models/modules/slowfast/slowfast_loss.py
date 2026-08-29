import torch
import torch.nn as nn
from ..criterions import SeqKD
from models.keys import Keys, require

class slowfast_loss(nn.Module):

    def __init__(self, loss_weights):
        super(slowfast_loss, self).__init__()
        self.loss = {}
        self.loss_weights = loss_weights
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)

    def forward(self, data):
        require(data, Keys.SEQUENCE_LOGITS, Keys.CONV_LOGITS, Keys.LABEL, Keys.FEAT_LEN, Keys.LABEL_LGT,
                who="slowfast_loss")
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'SeqCTC':
                total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](
                    data[Keys.SEQUENCE_LOGITS][0].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(), data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int()).mean()
                loss += total_loss['SeqCTC']
            elif k == 'Slow' or k == 'Fast':
                i = 1 if k == 'Slow' else 2
                if len(data[Keys.SEQUENCE_LOGITS]) <= i or len(data[Keys.CONV_LOGITS]) <= i:
                    continue
                total_loss[f'{k}'] = weight * self.loss_weights.get('SeqCTC', 1.0) * self.loss['CTCLoss'](
                    data[Keys.SEQUENCE_LOGITS][i].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(), data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int()).mean()
                loss += total_loss[f'{k}']
                if 'ConvCTC' in self.loss_weights:
                    total_loss[f'ConvCTC_{k}'] = weight * self.loss_weights['ConvCTC'] * self.loss['CTCLoss'](
                        data[Keys.CONV_LOGITS][i].log_softmax(-1),
                        data[Keys.LABEL].cpu().int(), data[Keys.FEAT_LEN].cpu().int(),
                        data[Keys.LABEL_LGT].cpu().int()).mean()
                    loss += total_loss[f'ConvCTC_{k}']
                if 'Dist' in self.loss_weights:
                    total_loss[f'Dist_{k}'] = weight * self.loss_weights['Dist'] * self.loss['distillation'](
                        data[Keys.CONV_LOGITS][i],
                        data[Keys.SEQUENCE_LOGITS][i].detach(),
                        use_blank=False)
                    loss += total_loss[f'Dist_{k}']
            elif k == 'ConvCTC':
                total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](
                    data[Keys.CONV_LOGITS][0].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(), data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int()).mean()
                loss += total_loss['ConvCTC']
            elif k == 'Dist':
                total_loss['Dist'] = weight * self.loss['distillation'](
                    data[Keys.CONV_LOGITS][0],
                    data[Keys.SEQUENCE_LOGITS][0].detach(),
                    use_blank=False)
                loss += total_loss['Dist']

        return {
            Keys.LOSS: loss,
            Keys.TOTAL_LOSS: total_loss
        }
