import torch
import torch.nn as nn
from ..criterions import SeqKD

class slowfast_loss(nn.Module):

    def __init__(self, loss_weights):
        super(slowfast_loss, self).__init__()
        self.loss = {}
        self.loss_weights = loss_weights
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)

    def forward(self, data):
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'SeqCTC':
                total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](
                    data["sequence_logits"][0].log_softmax(-1),
                    data["label"].cpu().int(), data["feat_len"].cpu().int(),
                    data["label_lgt"].cpu().int()).mean()
                loss += total_loss['SeqCTC']
            elif k == 'Slow' or k == 'Fast':
                i = 1 if k == 'Slow' else 2
                if len(data["sequence_logits"]) <= i or len(data["conv_logits"]) <= i:
                    continue
                total_loss[f'{k}'] = weight * self.loss_weights.get('SeqCTC', 1.0) * self.loss['CTCLoss'](
                    data["sequence_logits"][i].log_softmax(-1),
                    data["label"].cpu().int(), data["feat_len"].cpu().int(),
                    data["label_lgt"].cpu().int()).mean()
                loss += total_loss[f'{k}']
                if 'ConvCTC' in self.loss_weights:
                    total_loss[f'ConvCTC_{k}'] = weight * self.loss_weights['ConvCTC'] * self.loss['CTCLoss'](
                        data["conv_logits"][i].log_softmax(-1),
                        data["label"].cpu().int(), data["feat_len"].cpu().int(),
                        data["label_lgt"].cpu().int()).mean()
                    loss += total_loss[f'ConvCTC_{k}']
                if 'Dist' in self.loss_weights:
                    total_loss[f'Dist_{k}'] = weight * self.loss_weights['Dist'] * self.loss['distillation'](
                        data["conv_logits"][i],
                        data["sequence_logits"][i].detach(),
                        use_blank=False)
                    loss += total_loss[f'Dist_{k}']
            elif k == 'ConvCTC':
                total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](
                    data["conv_logits"][0].log_softmax(-1),
                    data["label"].cpu().int(), data["feat_len"].cpu().int(),
                    data["label_lgt"].cpu().int()).mean()
                loss += total_loss['ConvCTC']
            elif k == 'Dist':
                total_loss['Dist'] = weight * self.loss['distillation'](
                    data["conv_logits"][0],
                    data["sequence_logits"][0].detach(),
                    use_blank=False)
                loss += total_loss['Dist']

        return {
            "loss": loss,
            "total_loss": total_loss
        }
