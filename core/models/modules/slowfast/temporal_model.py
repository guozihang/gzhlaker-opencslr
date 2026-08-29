import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.slowfast.TemporalSlowFastConv1D import TemporalSlowFastConv1D
from models.keys import Keys, require

class temporal_model(nn.Module):
    def __init__(self, args, conv1d=None):
        super(temporal_model, self).__init__()
        #原来的temporal_model
        self.temporal_model = nn.ModuleList([
            BiLSTMLayer(
                rnn_type='LSTM',
                input_size=args.get("hidden_size", None),
                hidden_size=args.get("hidden_size", None),
                num_layers=2,
                bidirectional=True
            ) for i in range(3)
        ])
        # 原来的classifer
        weight_norm = args.get("weight_norm", None)
        share_classifier = args.get("share_classifier", None)
        self.num_classes = args.get("num_classes", None)
        self.hidden_size = args.get("hidden_size", None)
        self.conv1d = conv1d
        if weight_norm:
            self.classifier = nn.ModuleList([NormLinear(self.hidden_size, self.num_classes) for i in range(3)])
            if self.conv1d is not None:
                self.conv1d.conv1d.fc = nn.ModuleList(
                    [NormLinear(self.hidden_size, self.num_classes) for i in range(3)]
                )
        else:
            self.classifier = nn.ModuleList([nn.Linear(self.hidden_size, self.num_classes) for i in range(3)])
            if self.conv1d is not None:
                self.conv1d.conv1d.fc = nn.ModuleList(
                    [nn.Linear(self.hidden_size, self.num_classes) for i in range(3)]
                )
        if share_classifier == 1:
            if self.conv1d is not None:
                self.conv1d.conv1d.fc = self.classifier
        elif share_classifier == 2:
            classifier = self.classifier[0]
            self.classifier = nn.ModuleList([classifier for i in range(3)])
            if self.conv1d is not None:
                self.conv1d.conv1d.fc = nn.ModuleList([classifier for i in range(3)])

    def forward(self, data):

        require(data, Keys.VISUAL_FEAT, Keys.FEAT_LEN, who="temporal_model")
        outputs = []

        for i in range(len(data[Keys.VISUAL_FEAT])):
            tm_outputs = self.temporal_model[i](data[Keys.VISUAL_FEAT][i], data[Keys.FEAT_LEN])
            outputs.append(self.classifier[i](tm_outputs[Keys.PREDICTIONS]))


        return {
                Keys.SEQUENCE_LOGITS: outputs,
                Keys.OUTPUT_FIRST: outputs[0],
            }

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class BiLSTMLayer(nn.Module):
    def __init__(self, input_size, debug=False, hidden_size=512, num_layers=1, dropout=0.3,
                 bidirectional=True, rnn_type='LSTM', num_classes=-1):
        super(BiLSTMLayer, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.rnn_type = rnn_type
        self.debug = debug
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional)
        # for name, param in self.rnn.named_parameters():
        #     if name[:6] == 'weight':
        #         nn.init.orthogonal_(param)

    def forward(self, src_feats, src_lens, hidden=None):
        """
        Args:
            - src_feats: (max_src_len, batch_size, D)
            - src_lens: (batch_size)
        Returns:
            - outputs: (max_src_len, batch_size, hidden_size * num_directions)
            - hidden : (num_layers, batch_size, hidden_size * num_directions)
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
        """ If the encoder is bidirectional, do the following transformation.
            Ref: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176
            -----------------------------------------------------------
            In: (num_layers * num_directions, batch_size, hidden_size)
            (ex: num_layers=2, num_directions=2)

            layer 1: forward__hidden(1)
            layer 1: backward_hidden(1)
            layer 2: forward__hidden(2)
            layer 2: backward_hidden(2)

            -----------------------------------------------------------
            Out: (num_layers, batch_size, hidden_size * num_directions)

            layer 1: forward__hidden(1) backward_hidden(1)
            layer 2: forward__hidden(2) backward_hidden(2)
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
