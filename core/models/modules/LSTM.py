import torch.nn as nn
from .BiLSTM import BiLSTMLayer

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
        return self.temporal_model ( data [ 'visual_feat' ] , data [ 'feat_len' ] )