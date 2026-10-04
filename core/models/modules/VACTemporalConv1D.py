import torch.nn as nn
from .tconv import VACTemporalConv

class VACTemporalConv1D ( nn.Module ) :
    def __init__ ( self , args ) :
        super ( VACTemporalConv1D , self ).__init__ ( )
        self.conv1d = VACTemporalConv (
            input_size = 512 ,
            hidden_size = 1024 ,
            kernel_size = args["kernel_size"] ,
            stride = args["stride"] ,
            use_bn = args["use_bn"] ,
            num_classes = args["num_classes"] ,
        )

    def forward ( self , data) :
        return self.conv1d ( data['framewise_features'] , data['vid_lgt'] )