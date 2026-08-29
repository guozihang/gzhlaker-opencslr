import torch.nn as nn
from .norm import NormLinear
from models.keys import Keys, require
class Classifier ( nn.Module ) :
    def __init__ ( self , args ) :
        super ( Classifier , self ).__init__ ( )
        self.classifier = nn.Linear ( 1024 , args["num_classes"] )

    def forward ( self , data) :
        require ( data , Keys.PREDICTIONS , who = "Classifier" )
        return {
            Keys.SEQUENCE_LOGITS: self.classifier ( data[Keys.PREDICTIONS] )
        }