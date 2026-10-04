import torch
import torch.nn as nn
import torchvision.models as models
from .identity import Identity

class ResNet( nn.Module ) :
    def __init__ ( self, args) :
        super ( ResNet , self ).__init__ ( )
        c2d_type = args["c2d_type"]
        self.conv2d = getattr ( models , c2d_type ) ( weights = "ResNet18_Weights.IMAGENET1K_V1" )
        self.conv2d.fc = Identity ()

    def masked_bn ( self , inputs , len_x ) :
        def pad ( tensor , length ) :
            return torch.cat (
                [ tensor , tensor.new ( length - tensor.size ( 0 ) , *tensor.size ( ) [ 1 : ] ).zero_ ( ) ] )

        x = torch.cat ( [ inputs [ len_x [ 0 ] * idx :len_x [ 0 ] * idx + lgt ] for idx , lgt in enumerate ( len_x ) ] )
        x = self.conv2d ( x )
        x = torch.cat ( [ pad ( x [ sum ( len_x [ :idx ] ) :sum ( len_x [ :idx + 1 ] ) ] , len_x [ 0 ] )
                          for idx , lgt in enumerate ( len_x ) ] )
        return x

    def forward(self , data) :
        x = data['vid']
        len_x = data['vid_lgt']
        if len ( x.shape ) == 5 :
            # videos
            batch , temp , channel , height , width = x.shape
            inputs = x.reshape ( batch * temp , channel , height , width )
            framewise = self.masked_bn ( inputs , len_x )
            framewise = framewise.reshape ( batch , temp , -1 ).transpose ( 1 , 2 )
        else :
            # frame-wise features
            framewise = x
        return {
            "framewise_features": framewise,
            "visual_features" : x,
        }