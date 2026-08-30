# -*- coding: utf-8 -*-
"""模型构建工厂函数。

每个函数实例化一个完整的 SignLanguageModel,包含四个容器:
  - spatial_module_container:  空间特征提取
  - temporal_module_container: 时序建模与分类
  - loss_module_container:     损失计算
  - decoder:                   序列解码

在配置文件中通过 `model: models.build_function.build_xxx` 引用。
"""

from torch.nn.functional import conv1d

from models.base import Container, SignLanguageModel
from models.modules import ResNet, TemporalConv1D, BiLSTM, Classifier, Decoder, TLPLoss, VACLoss, VACTemporalConv1D
from models.modules.CorrNet_TemporalConv1D import CorrNeT_TemporalConv1D
from models.modules.corrnet_loss import CorrNetLoss
from models.modules.corrnet_resnet import corrnet_resnet18
from models.modules.norm import NormLinear
from models.senmodules.senresnet import SENresnet
from models.senmodules.SENLoss import SENLoss
from models.senmodules.sen_TemporalConv import sen_TemporalConv
from models.senmodules.sen_Decoder import sen_Decoder

from models.modules.slowfast.SlowFast import SlowFast
from models.modules.slowfast.TemporalSlowFastConv1D import TemporalSlowFastConv1D
from models.modules.slowfast.temporal_model import temporal_model
from models.modules.slowfast.slowfast_loss import slowfast_loss
from models.modules.slowfast.Decoder import SlowFast_Decoder

import torch.nn as nn


def build_tlp(args, gloss_dict, loss_weights):
    """构建 TLP (Temporal Lift Pooling) 模型。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 ResNet,时序模块为
        TemporalConv1D + BiLSTM + Classifier,损失为 TLPLoss。
    """
    return SignLanguageModel(
        spatial_module_container=Container([
            ResNet(args)
        ]),
        temporal_module_container=Container([
            TemporalConv1D(args),
            BiLSTM(args),
            Classifier(args)
        ]),
        loss_module_container=Container([
            TLPLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
    return model


def build_sen(args, gloss_dict, loss_weights):
    """构建 SEN (Stochastic Encoding Network) 模型。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 SENresnet,时序模块为
        sen_TemporalConv + BiLSTM + Classifier,损失为 SENLoss。
    """
    return SignLanguageModel(
        spatial_module_container=Container([
            SENresnet(args)
        ]),
        temporal_module_container=Container([
            sen_TemporalConv(args),
            BiLSTM(args),
            Classifier(args)
        ]),
        loss_module_container=Container([
            SENLoss(loss_weights)
        ]),
        decoder=sen_Decoder(args, gloss_dict)
    )


def build_vac(args, gloss_dict, loss_weights):
    """构建 VAC (Visual Alignment Constraint) 模型。

    注意:VAC 的时序卷积与分类器共享权重(NormLinear),且 conv1d 的 fc
    直接指向 classifier.classifier 的引用。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 ResNet,时序模块为
        VACTemporalConv1D + BiLSTM + Classifier,损失为 VACLoss。
    """
    conv1d = VACTemporalConv1D(args)
    classifier = Classifier(args)
    classifier.classifier = NormLinear(1024, args["num_classes"])
    conv1d.conv1d.fc = classifier.classifier
    return SignLanguageModel(
        spatial_module_container=Container([
            ResNet(args)
        ]),
        temporal_module_container=Container([
            conv1d,
            BiLSTM(args),
            classifier
        ]),
        loss_module_container=Container([
            VACLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
    return model


def build_slowfast(args, gloss_dict, loss_weights):
    """构建 SlowFast 模型。

    SlowFast 使用双路径空间网络(慢路径+快路径)和专用的时序卷积与解码器。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 SlowFast,时序模块为
        TemporalSlowFastConv1D + temporal_model,损失为 slowfast_loss。
    """
    conv1d = TemporalSlowFastConv1D(args)

    return SignLanguageModel(
        spatial_module_container=Container([
            SlowFast(args)
        ]),

        temporal_module_container=Container([
            conv1d,
            temporal_model(args, conv1d),
        ]),

        loss_module_container=Container([
            slowfast_loss(loss_weights)
        ]),

        decoder=SlowFast_Decoder(args, gloss_dict)
    )


def build_corrnet(args, gloss_dict, loss_weights):
    """构建 CorrNet (Correlation Network) 模型。

    CorrNet 支持可选的权重归一化(weight_norm)和分类器共享(share_classifier),
    当 share_classifier=True 时 conv1d 的 fc 与 classifier 的 classifier 共享参数。

    Args:
        args: 配置字典,包含模型超参数。
        gloss_dict: 词汇表字典,用于解码器映射。
        loss_weights: 损失权重列表,用于 CTC 损失。

    Returns:
        SignLanguageModel 实例,空间模块为 corrnet_resnet18,时序模块为
        CorrNeT_TemporalConv1D + BiLSTM + Classifier,损失为 CorrNetLoss。
    """
    conv1d = CorrNeT_TemporalConv1D(args)
    classifier = Classifier(args)
    hidden_size = args.get("hidden_size", 1024)
    num_classes = args["num_classes"]

    if args.get("weight_norm", True):
        classifier.classifier = NormLinear(hidden_size, num_classes)
        conv1d.conv1d.fc = NormLinear(hidden_size, num_classes)
    else:
        classifier.classifier = nn.Linear(hidden_size, num_classes)
        conv1d.conv1d.fc = nn.Linear(hidden_size, num_classes)

    if args.get("share_classifier", True):
        conv1d.conv1d.fc = classifier.classifier

    return SignLanguageModel(
        spatial_module_container=Container([
            corrnet_resnet18(args)
        ]),
        temporal_module_container=Container([
            conv1d,
            BiLSTM(args),
            classifier
        ]),
        loss_module_container=Container([
            CorrNetLoss(loss_weights)
        ]),
        decoder=Decoder(args, gloss_dict)
    )
