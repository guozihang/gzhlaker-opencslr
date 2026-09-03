"""SlowFast 的时序建模与分类模块。"""

import torch.nn as nn

from models.keys import Keys, require
from models.modules.BiLSTM import BiLSTMLayer
from models.modules.norm import NormLinear


class temporal_model(nn.Module):
    """使用三路 BiLSTM 对 SlowFast 特征建模并输出分类 logits。"""

    def __init__(self, args, conv1d=None):
        super(temporal_model, self).__init__()
        hidden_size = args.get("hidden_size", None)
        self.temporal_model = nn.ModuleList([
            BiLSTMLayer(
                rnn_type="LSTM",
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
            )
            for _ in range(3)
        ])

        weight_norm = args.get("weight_norm", None)
        share_classifier = args.get("share_classifier", None)
        self.num_classes = args.get("num_classes", None)
        self.hidden_size = hidden_size

        object.__setattr__(self, "_conv1d", conv1d)
        classifier_type = NormLinear if weight_norm else nn.Linear
        self.classifier = nn.ModuleList([
            classifier_type(self.hidden_size, self.num_classes) for _ in range(3)
        ])
        if self._conv1d is not None:
            self._conv1d.conv1d.fc = nn.ModuleList([
                classifier_type(self.hidden_size, self.num_classes) for _ in range(3)
            ])

        if share_classifier == 1:
            if self._conv1d is not None:
                self._conv1d.conv1d.fc = self.classifier
        elif share_classifier == 2:
            classifier = self.classifier[0]
            self.classifier = nn.ModuleList([classifier for _ in range(3)])
            if self._conv1d is not None:
                self._conv1d.conv1d.fc = nn.ModuleList([classifier for _ in range(3)])

    def forward(self, data):
        require(data, Keys.VISUAL_FEAT, Keys.FEAT_LEN, who="temporal_model")
        outputs = []
        for i in range(len(data[Keys.VISUAL_FEAT])):
            tm_outputs = self.temporal_model[i](
                data[Keys.VISUAL_FEAT][i], data[Keys.FEAT_LEN]
            )
            outputs.append(self.classifier[i](tm_outputs[Keys.PREDICTIONS]))

        return {
            Keys.SEQUENCE_LOGITS: outputs,
            Keys.OUTPUT_FIRST: outputs[0],
        }
