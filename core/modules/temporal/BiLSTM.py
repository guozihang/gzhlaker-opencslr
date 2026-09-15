"""BiLSTM 时序建模模块。

两层的分工(合并在一个文件里,因为后者只是前者的薄包装):

    BiLSTMLayer  裸的 nn.Module,用 pack_padded_sequence 处理变长序列,
                 产出 Keys.PREDICTIONS / Keys.HIDDEN。temporal_model(SlowFast)
                 也直接用它。
    BiLSTM       data dict 进出的包装层,固定成 2 层双向、hidden 1024。
"""

import torch
import torch.nn as nn
from models import Keys, require


class BiLSTMLayer(nn.Module):
    """Bidirectional LSTM layer for temporal modeling.

    Processes sequence features through a bidirectional LSTM (or GRU),
    handling variable-length sequences with packed padding.

    Args:
        input_size: Dimensionality of input features.
        hidden_size: Total hidden size (will be split if bidirectional).
        num_layers: Number of recurrent layers.
        dropout: Dropout rate between layers.
        bidirectional: Whether to use bidirectional RNN.
        rnn_type: Type of RNN cell ('LSTM' or 'GRU').
    """

    def __init__(self, input_size, hidden_size=512, num_layers=1, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM'):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)

    def forward(self, src_feats, src_lens, hidden=None):
        """Forward pass through the BiLSTM layer.

        Args:
            src_feats: Input features, shape (max_src_len, batch_size, D).
            src_lens: Actual lengths of each sequence, shape (batch_size,).
            hidden: Optional initial hidden state.

        Returns:
            dict: Contains:
                - Keys.PREDICTIONS: Output features, shape (max_src_len, batch_size, hidden_size * num_directions)
                - Keys.HIDDEN: Hidden state, shape (num_layers, batch_size, hidden_size * num_directions)
        """
        # (max_src_len, batch_size, D)
        packed_emb = nn.utils.rnn.pack_padded_sequence(src_feats, src_lens)

        # rnn(gru) returns:
        # - packed_outputs: shape same as packed_emb
        # - hidden: (num_layers * num_directions, batch_size, hidden_size)
        if hidden is not None and self.rnn_type == 'LSTM':
            half = int(hidden.size(0) / 2)
            hidden = (hidden[:half], hidden[half:])
        packed_outputs, hidden = self.rnn(packed_emb, hidden)

        # outputs: (max_src_len, batch_size, hidden_size * num_directions)
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            # (num_layers * num_directions, batch_size, hidden_size)
            # => (num_layers, batch_size, hidden_size * num_directions)
            hidden = self._cat_directions(hidden)

        if isinstance(hidden, tuple):
            # cat hidden and cell states
            hidden = torch.cat(hidden, 0)

        return {
            Keys.PREDICTIONS: rnn_outputs,
            Keys.HIDDEN: hidden
        }

    def _cat_directions(self, hidden):
        """Concatenate forward and backward hidden states.

        If the encoder is bidirectional, transform the hidden states
        from (num_layers * num_directions, batch_size, hidden_size) to
        (num_layers, batch_size, hidden_size * num_directions).

        Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176

        Args:
            hidden: Hidden state tensor or tuple of tensors (for LSTM).

        Returns:
            Concatenated hidden state with directions merged.
        """

        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)

        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)

        return hidden


class BiLSTM( nn.Module ) :
    """BiLSTM 时序建模模块。

    封装 BiLSTMLayer，配置为 2 层双向 LSTM，隐藏层大小为 1024，
    从数据字典中读取视觉特征并输出预测结果。

    Args:
        args: 配置字典（当前未使用，预留接口）。
    """

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
        """前向传播。

        从数据字典中提取视觉特征和序列长度，通过 BiLSTM 建模时序依赖。

        Args:
            data: 数据字典，需包含:
                Keys.VISUAL_FEAT: 视觉特征，形状 (T, B, C)
                Keys.FEAT_LEN: 每个序列的实际长度，形状 (B,)

        Returns:
            dict: 包含 Keys.PREDICTIONS 和 Keys.HIDDEN 的字典
        """
        require ( data , Keys.VISUAL_FEAT , Keys.FEAT_LEN , who = "BiLSTM" )
        return self.temporal_model ( data [ Keys.VISUAL_FEAT ] , data [ Keys.FEAT_LEN ] )
