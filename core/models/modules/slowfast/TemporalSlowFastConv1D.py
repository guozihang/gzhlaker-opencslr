# modules/temporal_slowfast_fuse.py
import copy
import torch
import torch.nn as nn
from models.keys import Keys, require

"""Temporal SlowFast 1D Convolution module.

Implements the temporal modeling component for the SlowFast architecture,
using 1D convolutions to fuse fast and slow pathways and model temporal
dependencies in sign language video sequences.
"""


class TemporalSlowFastConv1D( nn.Module ) :
    """Temporal SlowFast 1D Convolution wrapper.

    Wraps the TemporalSlowFastFuse module and applies three separate
    linear classifiers to the fused temporal features.

    Args:
        args: Configuration dictionary with hidden_size, conv_type, use_bn, num_classes.
    """

    def __init__ ( self , args ) :
        super ( TemporalSlowFastConv1D , self ).__init__ ( )
        hidden_size = args.get("hidden_size", None)
        self.conv1d = TemporalSlowFastFuse(
            fast_input_size=256, slow_input_size=2048, hidden_size=hidden_size,
            conv_type=args.get("conv_type", None), use_bn=args["use_bn"], num_classes=args["num_classes"] ,)

    def forward(self, data):

        """Forward pass of TemporalSlowFastConv1D.

        Args:
            data: Dict containing Keys.FRAMEWISE_FEATURES and Keys.VID_LGT.

        Returns:
            dict: Dict containing Keys.VISUAL_FEAT, Keys.FEAT_LEN, and Keys.CONV_LOGITS.
        """

        require(data, Keys.FRAMEWISE_FEATURES, Keys.VID_LGT, who="TemporalSlowFastConv1D")
        conv1d_outputs = self.conv1d(data[Keys.FRAMEWISE_FEATURES], data[Keys.VID_LGT])
        lgt = conv1d_outputs[Keys.FEAT_LEN]
        return{
            Keys.VISUAL_FEAT: conv1d_outputs[Keys.VISUAL_FEAT],
            Keys.FEAT_LEN: lgt,
            Keys.CONV_LOGITS: conv1d_outputs[Keys.CONV_LOGITS],
        }

class TemporalSlowFastFuse(nn.Module):
    """Temporal SlowFast feature fusion module.

    Applies separate 1D convolutions to the fast pathway, slow pathway,
    and fused (main) pathway features for temporal modeling, supporting
    multiple convolution configurations.

    Args:
        fast_input_size: Input feature dimension for the fast pathway.
        slow_input_size: Input feature dimension for the slow pathway.
        hidden_size: Hidden feature dimension.
        conv_type: Convolution type determining kernel configurations (0-10).
        use_bn: Whether to use batch normalization.
        num_classes: Number of output classes, -1 for no classifier.
    """

    def __init__(self, fast_input_size, slow_input_size, hidden_size, conv_type=2, use_bn=False, num_classes=-1):
        super(TemporalSlowFastFuse, self).__init__()
        self.use_bn = use_bn
        self.fast_input_size = fast_input_size
        self.slow_input_size = slow_input_size
        self.main_input_size = fast_input_size + slow_input_size
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
        elif self.conv_type == 9:
            self.kernel_size = ["K5", "K5", "P2"]
        elif self.conv_type == 10:
            self.kernel_size = ["K5", "K5"]

        fast_modules = []
        slow_modules = []
        main_modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            fast_input_sz = self.fast_input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            slow_input_sz = self.slow_input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            main_input_sz = self.main_input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                fast_modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                slow_modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
                main_modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                fast_modules.append(
                    nn.Conv1d(fast_input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                fast_modules.append(nn.BatchNorm1d(self.hidden_size))
                fast_modules.append(nn.ReLU(inplace=True))
                slow_modules.append(
                    nn.Conv1d(slow_input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                slow_modules.append(nn.BatchNorm1d(self.hidden_size))
                slow_modules.append(nn.ReLU(inplace=True))
                main_modules.append(
                    nn.Conv1d(main_input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                main_modules.append(nn.BatchNorm1d(self.hidden_size))
                main_modules.append(nn.ReLU(inplace=True))
        self.fast_temporal_conv = nn.Sequential(*fast_modules)
        self.slow_temporal_conv = nn.Sequential(*slow_modules)
        self.main_temporal_conv = nn.Sequential(*main_modules)

        if self.num_classes != -1:
            self.fc = nn.ModuleList([nn.Linear(self.hidden_size, self.num_classes) for i in range(3)])

    def update_lgt(self, lgt):
        """Update sequence lengths after convolution and pooling operations.

        Args:
            lgt: Original sequence lengths.

        Returns:
            Tensor: Updated sequence lengths.
        """
        feat_len = copy.deepcopy(lgt)
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
                # pass
        return feat_len

    def forward(self, frame_feat, lgt):
        """Forward pass of TemporalSlowFastFuse.

        Processes the fused frame features through separate temporal
        convolutions for the main pathway, slow pathway, and fast pathway.
        During training, all three pathways are computed; during evaluation,
        only the main pathway is used.

        Args:
            frame_feat: Frame-level features of shape (N, C, T).
            lgt: Sequence lengths for each sample.

        Returns:
            dict: Dict containing VISUAL_FEAT, CONV_LOGITS, and FEAT_LEN.
        """
        visual_feat = [self.main_temporal_conv(frame_feat)]
        if self.training:
            slow_path = frame_feat[:, :self.slow_input_size]
            fast_path = frame_feat[:, self.slow_input_size:]
            slow_feat = self.slow_temporal_conv(slow_path)
            fast_feat = self.fast_temporal_conv(fast_path)
            visual_feat.extend([slow_feat, fast_feat])
        num_paths = len(visual_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 \
            else [self.fc[i](visual_feat[i].transpose(1, 2)).transpose(1, 2) for i in range(num_paths)]
        return {
            Keys.VISUAL_FEAT: [visual_feat[i].permute(2, 0, 1) for i in range(num_paths)],
            Keys.CONV_LOGITS: [logits[i].permute(2, 0, 1) for i in range(num_paths)],
            Keys.FEAT_LEN: lgt.cpu(),
        }
