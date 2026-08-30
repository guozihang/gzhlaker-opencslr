"""CorrNet 时序卷积模块。

提供多尺度时序卷积和标准时序卷积，用于 CorrNet 模型中的时序特征建模。
"""

import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class MultiScale_TemporalConv(nn.Module):
    """多尺度时序卷积模块。

    使用多个不同膨胀率的并行卷积分支，分别处理后再拼接输出，
    捕获不同时间尺度的时序依赖关系。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        kernel_size: 卷积核大小（默认 3）。
        dilations: 各分支的膨胀率列表（默认 [1,2,3,4]）。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=[1,2,3,4],):

        super().__init__()

        # Multiple branches of temporal convolution
        self.num_branches = 4

        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels//self.num_branches,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding = dilation),
                nn.BatchNorm1d(out_channels//self.num_branches)
            )
            for dilation in dilations
        ])

        #self.fuse = nn.Conv1d(in_channels * self.num_branches, out_channels, kernel_size=1)
        #self.fuse = nn.Conv2d(in_channels, out_channels, kernel_size=(4,1))
        #self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """前向传播。

        通过多个膨胀卷积分支并行处理输入，拼接各分支输出。

        Args:
            x: 输入特征，形状 (N, C, T)

        Returns:
            torch.Tensor: 拼接后的多尺度特征，形状 (N, out_channels, T)
        """
        # Input dim: (N,C,T,V)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        #out = torch.stack(branch_outs, dim=2)
        #out = self.fuse(out).squeeze(2)
        #out = self.bn(out)
        return out


class TemporalConv(nn.Module):
    """CorrNet 时序卷积模块。

    根据 conv_type 配置构建不同结构的时序卷积网络，支持卷积和池化层的组合。
    可选最后的全连接分类头输出中间 logits。

    Args:
        input_size: 输入特征维度。
        hidden_size: 隐藏层特征维度。
        conv_type: 卷积类型（0-8），决定 kernel_size 配置。
        use_bn: 是否使用批归一化（当前未使用）。
        num_classes: 分类数，-1 表示不添加分类头。
    """

    def __init__(self, input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                    #MultiScale_TemporalConv(input_sz, self.hidden_size)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, lgt):
        """根据卷积和池化操作更新特征长度。

        Args:
            lgt: 原始特征长度

        Returns:
            torch.Tensor: 更新后的特征长度
        """
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                #pass
        return feat_len

    def forward(self, frame_feat, lgt):
        """前向传播。

        依次通过时序卷积层处理帧级特征，更新特征长度，
        并可选输出分类 logits。

        Args:
            frame_feat: 帧级特征，形状 (B, C, T)
            lgt: 每个样本的实际长度

        Returns:
            dict: 包含视觉特征、卷积 logits 和特征长度
        """
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            "visual_feat": visual_feat.permute(2, 0, 1),
            "conv_logits": logits.permute(2, 0, 1),
            "feat_len": lgt.cpu(),
        }
