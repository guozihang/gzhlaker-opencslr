import torch.nn as nn
from .tconv import VACTemporalConv
from models.keys import Keys, require

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
        require ( data , Keys.FRAMEWISE_FEATURES , Keys.VID_LGT , who = "VACTemporalConv1D" )
        return self.conv1d ( data[Keys.FRAMEWISE_FEATURES] , data[Keys.VID_LGT] )