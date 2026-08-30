# modules/SlowFast.py
import inspect
import os
import torch
import torch.nn as nn
import torchvision.models as models
from models.modules.slowfast.slowfast_modules import slowfast
from models.keys import Keys, require

"""SlowFast backbone for sign language recognition.

Implements the SlowFast video architecture as a spatial feature extractor,
supporting various C2D backbones (e.g., SlowFast, I3D) for video-based
sign language recognition.
"""


class SlowFast(nn.Module):
    """SlowFast backbone wrapper for video feature extraction.

    Wraps the SlowFast or C2D video backbone and processes video frames
    to extract frame-wise visual features. Supports both 5D video tensors
    and pre-extracted features.

    Args:
        args: Configuration dictionary containing model parameters such as
              c2d_type, slowfast_config, slowfast_args, and num_classes.
    """

    def __init__(self, args):
        super(SlowFast, self).__init__()

        c2d_type = args.get("c2d_type", None)
        conv_type = args.get("conv_type", None)
        slowfast_config = args.get("slowfast_config", None)
        slowfast_args = args.get("slowfast_args", None)
        load_pkl = args.get("load_pkl", True)
        num_classes = args.get("num_classes", None)

        builder = getattr(slowfast, c2d_type)
        signature = inspect.signature(builder)
        builder_kwargs = {}
        slowfast_path = os.path.join(os.path.dirname(os.path.abspath(slowfast.__file__)))

        if "slowfast_config" in signature.parameters and slowfast_config is not None:
            builder_kwargs["slowfast_config"] = slowfast_config
        if "slowfast_args" in signature.parameters and slowfast_args is not None:
            builder_kwargs["slowfast_args"] = slowfast_args
        if "slowfast_path" in signature.parameters:
            builder_kwargs["slowfast_path"] = slowfast_path
        if "load_pkl" in signature.parameters:
            builder_kwargs["load_pkl"] = load_pkl
        if "multi" in signature.parameters:
            builder_kwargs["multi"] = True

        self.conv2d = builder(**builder_kwargs)

    def masked_bn(self, inputs, len_x):
        """Apply masked batch normalization with variable-length inputs.

        Handles batches where each sample has a different temporal length
        by applying the backbone to each sample individually and padding
        the results back to uniform length.

        Args:
            inputs: Input tensor.
            len_x: List of lengths for each sample in the batch.

        Returns:
            Tensor: Output after batch normalization, padded to uniform length.
        """
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, data):
        """Forward pass of SlowFast backbone.

        Extracts frame-wise features from video input. Supports both
        raw video tensors (5D with shape N,C,T,H,W) and pre-extracted
        features.

        Args:
            data: Dict containing Keys.VID with video tensor.

        Returns:
            dict: Dict containing Keys.FRAMEWISE_FEATURES.
        """
        require(data, Keys.VID, who="SlowFast")
        x = data[Keys.VID]

        if len(x.shape) == 5:
            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))
        else:
            # frame-wise features
            framewise = x
        return {
            Keys.FRAMEWISE_FEATURES: framewise,
        }
