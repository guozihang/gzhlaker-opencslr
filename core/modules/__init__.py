# -*- coding: utf-8 -*-
"""共用积木模块。

这里只做汇总导出,方便 models/ 下的具体模型一行声明自己的组装;实现按用途
分子目录存放——**四类目录里只放各自相关的东西**,不属于任何一类的辅助积木
统一放 others/:

    spatio/             空间网络: ResNet, SENresnet, corrnet_resnet18,
                                  SlowFast(含 vendored slowfast_modules/)
    temporal/           时序网络: TemporalConv1D, VACTemporalConv, BiLSTM
                                  (含裸层 BiLSTMLayer), CorrNeT_TemporalConv1D,
                                  TemporalSlowFastConv1D, temporal_model
    losses/             最小单元损失: CTCLoss, SeqKD
                        模型专有的损失组装写在各模型文件里
    decoders/           解码:     Decoder(SlowFast 的子类在其模型文件里)
    others/             辅助积木: Identity(换掉骨干的 fc)、Classifier(分类头)、
                                  NormLinear(分类头共用)、TemporalLiftPooling
                                  (带可学习参数的池化,含配套的 Local_Weighting)
"""

# 空间
from .spatio.ResNet import ResNet
from .spatio.senresnet import SENresnet
from .spatio.corrnet_resnet import corrnet_resnet18
from .spatio.SlowFast import SlowFast

# 时序
from .temporal.BiLSTM import BiLSTM
from .temporal.TemporalConv1D import TemporalConv1D
from .temporal.CorrNet_TemporalConv1D import CorrNeT_TemporalConv1D
from .temporal.tconv import VACTemporalConv
from .temporal.TemporalSlowFastConv1D import TemporalSlowFastConv1D
from .temporal.temporal_model import temporal_model

# 最小单元损失
from .losses.ctc import CTCLoss
from .losses.kd import SeqKD

# 解码
from .decoders.decoder import Decoder

# 辅助积木
from .others.classifier import Classifier
from .others.identity import Identity
from .others.liftpool import TemporalLiftPooling
from .others.norm import NormLinear
