from torch.nn.functional import conv1d

from models.base import Container , SignLanguageModel
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
    return SignLanguageModel(
        spatial_module_container = Container([
            ResNet(args)
        ]),
        temporal_module_container = Container([
            TemporalConv1D(args),
            BiLSTM(args),
            Classifier ( args )
        ]),
        loss_module_container = Container([
            TLPLoss(loss_weights)
        ]),
        decoder = Decoder(args, gloss_dict)
    )
    return model

def build_sen(args, gloss_dict, loss_weights):
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
    conv1d = VACTemporalConv1D ( args )
    classifier = Classifier( args )
    classifier.classifier = NormLinear(1024, args["num_classes"])
    conv1d.conv1d.fc = classifier.classifier
    return SignLanguageModel (
        spatial_module_container = Container ( [
            ResNet (args)
        ] ) ,
        temporal_module_container = Container ( [
            conv1d,
            BiLSTM ( args ) ,
            classifier
        ] ) ,
        loss_module_container = Container ( [
            VACLoss ( loss_weights )
        ] ) ,
        decoder = Decoder ( args , gloss_dict )
    )
    return model

def build_slowfast(args, gloss_dict, loss_weights):
    conv1d = TemporalSlowFastConv1D(args)

    return SignLanguageModel(
        spatial_module_container=Container([
            SlowFast(args)
        ]),

        temporal_module_container=Container([
            conv1d,
            temporal_model(args, conv1d),
            # slowfast_classifier(args)
        ]),

        loss_module_container=Container([
            slowfast_loss(loss_weights)
        ]),

        decoder=SlowFast_Decoder(args, gloss_dict)
    )

def build_corrnet(args, gloss_dict, loss_weights):
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
