# modules/norm_linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.slowfast.TemporalSlowFastConv1D import TemporalSlowFastConv1D


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class slowfast_classifier(nn.Module):
    def __init__(self, args):
        super(slowfast_classifier, self).__init__()
        weight_norm = args.get("weight_norm", None)
        share_classifier = args.get("share_classifier", None)
        self.num_classes = args.get("num_classes", None)
        self.conv1d = TemporalSlowFastConv1D(args)

        if weight_norm:
            self.classifier = nn.ModuleList([NormLinear(1024, self.num_classes) for i in range(3)])
            self.conv1d.fc = nn.ModuleList([NormLinear(1024, self.num_classes) for i in range(3)])
        else:
            self.classifier = nn.ModuleList([nn.Linear(1024, self.num_classes) for i in range(3)])
            self.conv1d.fc = nn.ModuleList([nn.Linear(1024, self.num_classes) for i in range(3)])

        if share_classifier == 1:
            self.conv1d.fc = self.classifier
        elif share_classifier == 2:
            classifier = self.classifier[0]
            self.classifier = nn.ModuleList([classifier for i in range(3)])
            self.conv1d.fc = nn.ModuleList([classifier for i in range(3)])

    def forward(self, data):
        print()
        print(f"slowfast_classifer_debug:data keys: {data.keys()}")

        predictions = data['predictions']
        print(f"predictions type: {type(predictions)}")

        # 如果 predictions 是列表, 对每个预测执行分类器
        if isinstance(predictions, list):
            sequence_logits = [self.classifier[i](pred) for i, pred in enumerate(predictions)]
        else:
            # 如果 predictions 不是列表, 直接使用第一个分类器
            sequence_logits = self.classifier[0](predictions)

        return {
            "sequence_logits": sequence_logits
        }
