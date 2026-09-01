import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.slowfast.TemporalSlowFastConv1D import TemporalSlowFastConv1D
from models.keys import Keys, require

"""SlowFast 时序模型模块。

使用双向 LSTM 对 SlowFast 提取的特征进行时序建模，
并支持权重归一化和分类器共享机制。
"""


class temporal_model(nn.Module):
    """SlowFast 时序模型。

    使用三个独立的 BiLSTM 层分别对 SlowFast 的三个路径（主路径、慢路径、快路径）
    进行时序建模，并可选地使用权重归一化或分类器共享机制。

    Args:
        args: 配置字典，包含 hidden_size, num_classes, weight_norm, share_classifier 等
        conv1d: 可选的 TemporalSlowFastConv1D 模块，用于替换分类器
    """

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
        # 用 object.__setattr__ 持有 conv1d 引用,避免同一模块被重复注册到 state_dict
        object.__setattr__(self, '_conv1d', conv1d)
        if weight_norm:
            self.classifier = nn.ModuleList([NormLinear(self.hidden_size, self.num_classes) for i in range(3)])
            if self._conv1d is not None:
                self._conv1d.conv1d.fc = nn.ModuleList(
                    [NormLinear(self.hidden_size, self.num_classes) for i in range(3)]
                )
        else:
            self.classifier = nn.ModuleList([nn.Linear(self.hidden_size, self.num_classes) for i in range(3)])
            if self._conv1d is not None:
                self._conv1d.conv1d.fc = nn.ModuleList(
                    [nn.Linear(self.hidden_size, self.num_classes) for i in range(3)]
                )
        if share_classifier == 1:
            if self._conv1d is not None:
                self._conv1d.conv1d.fc = self.classifier
        elif share_classifier == 2:
            classifier = self.classifier[0]
            self.classifier = nn.ModuleList([classifier for i in range(3)])
            if self._conv1d is not None:
                self._conv1d.conv1d.fc = nn.ModuleList([classifier for i in range(3)])

    def forward(self, data):

        """前向传播。

        Args:
            data: 包含 Keys.VISUAL_FEAT 和 Keys.FEAT_LEN 的字典

        Returns:
            dict: 包含 SEQUENCE_LOGITS 和 OUTPUT_FIRST 的字典
        """

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
    """权重归一化的线性层。

    使用 L2 归一化对权重矩阵进行归一化处理，
    然后执行矩阵乘法，替代标准的线性变换。

    Args:
        in_dim: 输入特征维度
        out_dim: 输出特征维度
    """

    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入张量，形状为 (..., in_dim)

        Returns:
            Tensor: 输出张量，形状为 (..., out_dim)
        """
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class BiLSTMLayer(nn.Module):
    """双向 LSTM 层。

    使用双向 LSTM/GRU 对序列数据进行时序建模，
    支持变长序列处理（通过 pack_padded_sequence）。

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层特征维度
        num_layers: LSTM 层数
        dropout: Dropout 概率
        bidirectional: 是否使用双向 LSTM
        rnn_type: RNN 类型（'LSTM' 或 'GRU'）
        num_classes: 分类数，-1 表示不输出分类结果
        debug: 是否启用调试模式
    """

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
        """拼接双向 RNN 的隐藏状态。

        将双向 RNN 的 forward 和 backward 隐藏状态沿特征维度拼接。
        参考：https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py#L176

        Args:
            hidden: 输入隐藏状态，形状为 (num_layers * num_directions, batch_size, hidden_size)

        Returns:
            Tensor 或 tuple: 拼接后的隐藏状态，形状为 (num_layers, batch_size, hidden_size * num_directions)
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
