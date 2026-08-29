import torch
from torch import nn
from ..modules.criterions import SeqKD
from ..keys import Keys, require
class SENLoss(nn.Module):
    def __init__(self,loss_weights):
        super(SENLoss,self).__init__()
        self.loss={}
        self.loss_weights=loss_weights
        self.loss['CTCLoss']=nn.CTCLoss(reduction='none',zero_infinity=False)
        self.loss['Distillation']=SeqKD(T=8)
    def forward(self,data):
        require(data, Keys.CONV_LOGITS, Keys.SEQUENCE_LOGITS, Keys.LABEL, Keys.FEAT_LEN, Keys.LABEL_LGT,
                who="SENLoss")
        loss=0
        total_loss={}
        for k,weight in self.loss_weights.items():
            if k=='ConvCTC':
                total_loss['ConvCTC']=weight*self.loss['CTCLoss'](
                    data[Keys.CONV_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int()
                ).mean()
                loss+=total_loss['ConvCTC']
            elif k=='SeqCTC':
                total_loss['SeqCTC']=weight*self.loss['CTCLoss'](
                    data[Keys.SEQUENCE_LOGITS].log_softmax(-1),
                    data[Keys.LABEL].cpu().int(),
                    data[Keys.FEAT_LEN].cpu().int(),
                    data[Keys.LABEL_LGT].cpu().int()
                ).mean()
                loss+=total_loss['SeqCTC']
            elif k=='Dist':
                total_loss['Dist']=weight*self.loss['Distillation'](
                    data[Keys.CONV_LOGITS],data[Keys.SEQUENCE_LOGITS].detach(),use_blank=False
                )
                loss+=total_loss['Dist']
        return {
            Keys.LOSS:loss,
            Keys.TOTAL_LOSS:total_loss
        }
