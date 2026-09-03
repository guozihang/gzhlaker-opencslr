"""SlowFast 多路径分类器。"""

import torch.nn as nn

from models.keys import Keys, require
from models.modules.norm import NormLinear
from models.modules.slowfast.TemporalSlowFastConv1D import TemporalSlowFastConv1D


class slowfast_classifier(nn.Module):
    """为 SlowFast 的三路时序特征提供线性分类器。"""

    def __init__(self, args):
        super(slowfast_classifier, self).__init__()
        weight_norm = args.get("weight_norm", None)
        share_classifier = args.get("share_classifier", None)
        self.num_classes = args.get("num_classes", None)
        self.conv1d = TemporalSlowFastConv1D(args)

        classifier_type = NormLinear if weight_norm else nn.Linear
        self.classifier = nn.ModuleList([
            classifier_type(1024, self.num_classes) for _ in range(3)
        ])
        self.conv1d.fc = nn.ModuleList([
            classifier_type(1024, self.num_classes) for _ in range(3)
        ])

        if share_classifier == 1:
            self.conv1d.fc = self.classifier
        elif share_classifier == 2:
            classifier = self.classifier[0]
            self.classifier = nn.ModuleList([classifier for _ in range(3)])
            self.conv1d.fc = nn.ModuleList([classifier for _ in range(3)])

    def forward(self, data):
        require(data, Keys.PREDICTIONS, who="slowfast_classifier")
        predictions = data[Keys.PREDICTIONS]
        if isinstance(predictions, list):
            sequence_logits = [
                self.classifier[i](pred) for i, pred in enumerate(predictions)
            ]
        else:
            sequence_logits = self.classifier[0](predictions)
        return {Keys.SEQUENCE_LOGITS: sequence_logits}
