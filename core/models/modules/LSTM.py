import torch.nn as nn
from .BiLSTM import BiLSTMLayer
from models.keys import Keys, require

class BiLSTM( nn.Module ) :
    def __init__ ( self , args) :
        super ( BiLSTM , self ).__init__ ( )
        self.temporal_model = BiLSTMLayer (
            rnn_type = 'LSTM' ,
            input_size = 1024 ,
            hidden_size = 1024 ,
            num_layers = 2 ,
            bidirectional = True
        )

    def forward ( self , data) :
        require ( data , Keys.VISUAL_FEAT , Keys.FEAT_LEN , who = "BiLSTM" )
        return self.temporal_model ( data [ Keys.VISUAL_FEAT ] , data [ Keys.FEAT_LEN ] )