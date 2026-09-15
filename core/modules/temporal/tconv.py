"""时序卷积模块，用于 CSLR 模型中的时序特征建模。

提供 TemporalConv（TLP/SEN 用，'P<n>' 层是 LiftPool）和 VACTemporalConv
（VAC 用，'P<n>' 层是标准 MaxPool1d）两个时序处理模块。
LiftPool 本身在 modules/others/liftpool.py。
"""

import copy
import torch
import torch.nn as nn
from models import Keys

from modules.others.liftpool import TemporalLiftPooling


class TemporalConv(nn.Module):
    """标准时序卷积模块，支持 LiftPool 和 Conv1D 混合架构。

    根据 kernel_size 配置构建多层时序处理网络，每层可以是
    标准卷积（'K'）或提升池化（'P'），可选最后的全连接分类头。

    Args:
        input_size: 输入特征维度。
        hidden_size: 隐藏层特征维度。
        kernel_size: 每层的配置列表，如 ['K3', 'P2', 'K5']。
        stride: 步长列表（当前仅保留接口兼容）。
        use_bn: 是否使用批归一化（当前未使用）。
        num_classes: 分类数，-1 表示不添加分类头。
    """

    def __init__(self, input_size, hidden_size, kernel_size=['K3'], stride=[0], use_bn=False, num_classes=-1):
        super(TemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.strides = stride

        # kernel_size 里 'K<n>' 是 n×1 卷积, 'P<n>' 是 LiftPool 下采样
        self.temporal_conv = nn.ModuleList([])
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                self.temporal_conv.append(TemporalLiftPooling(input_size=input_sz, kernel_size=int(ks[1])))
            elif ks[0] == 'K':
                self.temporal_conv.append(
                    nn.Sequential(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.ReLU(inplace=True),
                    )
                )

        if self.num_classes != -1:
            self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def update_lgt(self, feat_len):
        """根据卷积和池化操作更新特征长度。

        遍历 kernel_size 配置，对每层操作计算输出长度：
        'P'（池化）: 长度除以核大小
        'K'（卷积）: 长度减去 (kernel_size - 1)

        Args:
            feat_len: 原始特征长度

        Returns:
            torch.Tensor: 更新后的特征长度
        """
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len //= int(ks[1])
            else:
                feat_len -= int(ks[1]) - 1
        return feat_len

    def forward(self, frame_feat, lgt):
        """前向传播。

        依次通过时序卷积层处理帧级特征，累积 LiftPool 损失，
        更新特征长度，并可选输出分类 logits。

        Args:
            frame_feat: 帧级特征，形状 (B, C, T)
            lgt: 每个样本的实际长度

        Returns:
            dict: 包含视觉特征、卷积 logits、特征长度和 LiftPool 损失
        """
        loss_LiftPool_u = 0
        loss_LiftPool_p = 0
        i = 0
        visual_feat = frame_feat
        for tempconv in self.temporal_conv:
            if isinstance(tempconv, TemporalLiftPooling):
                visual_feat, loss_u, loss_d = tempconv(visual_feat) #self.strides[i])
                i +=1
                loss_LiftPool_u += loss_u
                loss_LiftPool_p += loss_d
            else:
                visual_feat = tempconv(visual_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            Keys.VISUAL_FEAT: visual_feat.permute(2, 0, 1),
            Keys.CONV_LOGITS: logits.permute(2, 0, 1),
            Keys.FEAT_LEN: lgt.cpu(),
            Keys.LOSS_LIFTPOOL_U: loss_LiftPool_u,
            Keys.LOSS_LIFTPOOL_P: loss_LiftPool_p,
        }

class VACTemporalConv(nn.Module):
    """VAC 模型的时序卷积模块。

    与 TemporalConv 类似，但使用标准 MaxPool1d 替代 LiftPool，
    适用于 VAC（Visual Alignment Constraint）模型架构。

    Args:
        input_size: 输入特征维度。
        hidden_size: 隐藏层特征维度。
        kernel_size: 每层的配置列表，如 ['K3', 'P2', 'K5']。
        stride: 步长列表（当前仅保留接口兼容）。
        use_bn: 是否使用批归一化（当前未使用）。
        num_classes: 分类数，-1 表示不添加分类头。
    """

    def __init__(self, input_size, hidden_size, kernel_size=['K3'], stride=[0], use_bn=False, num_classes=-1):
        super(VACTemporalConv, self).__init__()
        self.use_bn = use_bn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.strides = stride

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
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
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            Keys.VISUAL_FEAT: visual_feat.permute(2, 0, 1),
            Keys.CONV_LOGITS: logits.permute(2, 0, 1),
            Keys.FEAT_LEN: lgt.cpu(),
        }
