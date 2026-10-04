import torch.nn as nn
from .corrnet_tconv import TemporalConv

class CorrNeT_TemporalConv1D(nn.Module):
    def __init__(self, args):
        super(CorrNeT_TemporalConv1D, self).__init__()
        self.conv1d = TemporalConv(
            input_size=args.get("input_size", 512),
            hidden_size=args.get("hidden_size", 1024),
            conv_type=args.get("conv_type", 2),
            use_bn=args.get("use_bn", False),
            num_classes=args.get("num_classes", -1),
        )

    def forward(self, data):
        output = self.conv1d(data['framewise_features'], data['vid_lgt'])
        return output