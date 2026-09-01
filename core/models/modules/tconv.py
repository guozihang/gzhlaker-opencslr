"""时序卷积模块，用于 CSLR 模型中的时序特征建模。

提供 Temporal_LiftPool、TemporalConv 和 VACTemporalConv 等时序处理模块，
支持标准卷积和提升池化（LiftPool）操作。
"""

import copy
import pdb
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from models.keys import Keys


class Temporal_LiftPool(nn.Module):
    """时序提升池化（LiftPool）模块。

    基于提升方案（Lifting Scheme）的时序池化操作，通过预测器（Predictor）
    和更新器（Updater）自适应地分解和池化时序特征。

    Args:
        input_size: 输入特征通道数。
        kernel_size: 池化核大小（间隔），默认为2。
    """

    def __init__(self, input_size, kernel_size=2):
        super(Temporal_LiftPool, self).__init__()
        self.kernel_size = kernel_size
        self.predictor = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
                                    )

        self.updater = nn.Sequential(
            nn.Conv1d(input_size, input_size, kernel_size=3, stride=1, padding=1, groups=input_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
                                    )
        self.predictor[2].weight.data.fill_(0.0)
        self.updater[2].weight.data.fill_(0.0)
        self.weight1 = Local_Weighting(input_size)
        self.weight2 = Local_Weighting(input_size)

    def forward(self, x):
        """前向传播。

        将输入特征分解为偶数和奇数子序列，通过预测-更新机制实现提升池化。

        Args:
            x: 输入特征，形状 (B, C, T)

        Returns:
            tuple: (池化后的特征, 更新损失 u_loss, 预测损失 p_loss)
        """
        B, C, T= x.size()
        Xe = x[:,:,:T:self.kernel_size]
        Xo = x[:,:,1:T:self.kernel_size]
        d = Xo - self.predictor(Xe)
        s = Xe + self.updater(d)
        loss_u = torch.norm(s-Xo, p=2)
        loss_p = torch.norm(d, p=2)
        s = torch.cat((x[:,:,:0:self.kernel_size], s, x[:,:,T::self.kernel_size]),2)
        return self.weight1(s)+self.weight2(d), loss_u, loss_p


class Local_Weighting(nn.Module):
    """局部加权模块。

    通过卷积和实例归一化对特征进行局部自适应加权，
    输出为原始特征与加权调整值的组合。

    Args:
        input_size: 输入特征通道数。
    """

    def __init__(self, input_size ):
        super(Local_Weighting, self).__init__()
        self.conv = nn.Conv1d(input_size, input_size, kernel_size=5, stride=1, padding=2)
        self.insnorm = nn.InstanceNorm1d(input_size, affine=True)
        self.conv.weight.data.fill_(0.0)

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入特征，形状 (B, C, T)

        Returns:
            torch.Tensor: 加权后的特征，形状与输入相同
        """
        out = self.conv(x)
        return x + x*(F.sigmoid(self.insnorm(out))-0.5)


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

        # if self.conv_type == 0:
        #     self.kernel_size = ['K3']
        # elif self.conv_type == 1:
        #     self.kernel_size = ['K5', "P2"]
        #     self.strides = [0]
        # elif self.conv_type == 2:
        #     self.kernel_size =



        self.temporal_conv = nn.ModuleList([])
        #nums = 0
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                #nums += 1
                #if nums == 2:
                #    self.temporal_conv.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                #elif nums == 1:
                self.temporal_conv.append(Temporal_LiftPool(input_size=input_sz, kernel_size=int(ks[1])))
                #self.temporal_conv.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                #self.temporal_conv.append(nn.AvgPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                
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
            if isinstance(tempconv, Temporal_LiftPool):
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
