import torch
import torch.nn as nn
from .criterions import SeqKD

class VACLoss( nn.Module ) :
    def __init__ ( self, loss_weights) :
        super ( VACLoss , self ).__init__ ( )
        self.loss = {}
        self.loss_weights = loss_weights
        self.loss [ 'CTCLoss' ] = torch.nn.CTCLoss ( reduction = 'none' , zero_infinity = False )
        self.loss [ 'distillation' ] = SeqKD ( T = 8 )

    def forward ( self , data ) :
        loss = 0
        total_loss = {}
        for k , weight in self.loss_weights.items ( ) :
            if k == 'ConvCTC' :
                total_loss [ 'ConvCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    data [ "conv_logits" ].log_softmax ( -1 ) ,
                    data["label"].cpu ( ).int ( ) , data [ "feat_len" ].cpu ( ).int ( ) ,
                    data["label_lgt"].cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'ConvCTC' ]
            elif k == 'SeqCTC' :
                total_loss [ 'SeqCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    data [ "sequence_logits" ].log_softmax ( -1 ) ,
                    data["label"].cpu ( ).int ( ) , data [ "feat_len" ].cpu ( ).int ( ) ,
                    data["label_lgt"].cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'SeqCTC' ]
            elif k == 'Dist' :
                total_loss [ 'Dist' ] = weight * self.loss [ 'distillation' ] ( data [ "conv_logits" ] ,
                                                                                data [
                                                                                    "sequence_logits" ].detach ( ) ,
                                                                                use_blank = False )
                loss += total_loss [ 'Dist' ]
        return {
            "loss": loss,
            "total_loss": total_loss
        }

