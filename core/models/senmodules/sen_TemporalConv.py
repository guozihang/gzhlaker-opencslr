import copy

import torch
import torch.nn as nn


class SENTemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1, kernel_size=None):
        super(SENTemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.kernel_size = self._resolve_kernel_size(conv_type, kernel_size)

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0))
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
            else:
                raise ValueError("Unsupported SEN temporal kernel: {}".format(ks))

        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    @staticmethod
    def _resolve_kernel_size(conv_type, kernel_size):
        if kernel_size is not None:
            return kernel_size
        if conv_type == 0:
            return ['K3']
        if conv_type == 1:
            return ['K5', 'P2']
        if conv_type == 2:
            return ['K5', 'P2', 'K5', 'P2']
        raise ValueError("Unsupported SEN conv_type: {}".format(conv_type))

    def update_lgt(self, lgt):
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                if torch.is_tensor(feat_len):
                    feat_len = torch.div(feat_len, int(ks[1]), rounding_mode='floor')
                else:
                    feat_len //= int(ks[1])
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": None if logits is None else logits.permute(2, 0, 1),
            "feat_len": lgt.cpu() if torch.is_tensor(lgt) else lgt,
        }


class sen_TemporalConv(nn.Module):
    def __init__(self, args):
        super(sen_TemporalConv, self).__init__()
        use_config_kernel = "conv_type" not in args
        self.conv1d = SENTemporalConv(
            input_size=args.get("input_size", 512),
            hidden_size=args.get("hidden_size", 1024),
            conv_type=args.get("conv_type", 2),
            use_bn=args.get("use_bn", False),
            num_classes=args["num_classes"],
            kernel_size=args.get("kernel_size") if use_config_kernel else None,
        )

    def forward(self, data):
        return self.conv1d(data['framewise_features'], data['vid_lgt'])
