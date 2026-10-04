import torch
from torch import nn
from ..modules.criterions import SeqKD
class SENLoss(nn.Module):
    def __init__(self,loss_weights):
        super(SENLoss,self).__init__()
        self.loss={}
        self.loss_weights=loss_weights
        self.loss['CTCLoss']=nn.CTCLoss(reduction='none',zero_infinity=False)
        self.loss['Distillation']=SeqKD(T=8)
    def forward(self,data):
        loss=0
        total_loss={}
        for k,weight in self.loss_weights.items():
            if k=='ConvCTC':
                total_loss['ConvCTC']=weight*self.loss['CTCLoss'](
                    data['conv_logits'].log_softmax(-1),
                    data['label'].cpu().int(),
                    data['feat_len'].cpu().int(),
                    data["label_lgt"].cpu().int()
                ).mean()
                loss+=total_loss['ConvCTC']
            elif k=='SeqCTC':
                total_loss['SeqCTC']=weight*self.loss['CTCLoss'](
                    data['sequence_logits'].log_softmax(-1),
                    data['label'].cpu().int(),
                    data['feat_len'].cpu().int(),
                    data['label_lgt'].cpu().int()
                ).mean()
                loss+=total_loss['SeqCTC']
            elif k=='Dist':
                total_loss['Dist']=weight*self.loss['Distillation'](
                    data['conv_logits'],data['sequence_logits'].detach(),use_blank=False
                )
                loss+=total_loss['Dist']
        return {
            "loss":loss,
            "total_loss":total_loss
        }
