import torch.nn as nn
from .tconv import TemporalConv
from models.keys import Keys, require

class TemporalConv1D ( nn.Module ) :
    def __init__ ( self , args ) :
        super ( TemporalConv1D , self ).__init__ ( )
        self.conv1d = TemporalConv (
            input_size = 512 ,
            hidden_size = 1024 ,
            kernel_size = args["kernel_size"] ,
            stride = args["stride"] ,
            use_bn = args["use_bn"] ,
            num_classes = args["num_classes"] ,
        )

    def forward ( self , data) :
        require ( data , Keys.FRAMEWISE_FEATURES , Keys.VID_LGT , who = "TemporalConv1D" )
        return self.conv1d ( data[Keys.FRAMEWISE_FEATURES] , data[Keys.VID_LGT] )