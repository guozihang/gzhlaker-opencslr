"""SEN 模型的时间卷积模块。

提供基于 1D 卷积的时间建模功能，支持多种卷积配置，
并可选择性地输出分类 logits。
"""
import copy

import torch
import torch.nn as nn

from ..keys import Keys, require


class SENTemporalConv(nn.Module):
    """SEN 时间卷积模块。

    使用 1D 卷积对帧级特征进行时序建模，支持多种卷积核配置
    和可选的最大池化下采样，并可选择性地输出分类 logits。

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏层特征维度
        conv_type: 卷积类型 (0, 1, 2)，决定卷积核配置
        use_bn: 是否使用批归一化
        num_classes: 分类数，-1 表示不输出 logits
        kernel_size: 自定义卷积核配置，若为 None 则根据 conv_type 自动生成
    """

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
        """解析卷积核配置。

        Args:
            conv_type: 卷积类型 (0, 1, 2)
            kernel_size: 自定义卷积核配置，若不为 None 则直接返回

        Returns:
            list: 卷积核配置列表，每个元素为 ['K<size>'] 或 ['P<size>']
        """
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
        """根据卷积和池化操作更新序列长度。

        Args:
            lgt: 原始序列长度

        Returns:
            Tensor 或 int: 更新后的序列长度
        """
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
        """前向传播。

        Args:
            frame_feat: 帧级特征，形状为 (N, C, T)
            lgt: 每个样本的原始序列长度

        Returns:
            dict: 包含视觉特征、卷积 logits 和特征长度的字典
        """
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        logits = None if self.num_classes == -1 else self.fc(visual_feat.transpose(1, 2)).transpose(1, 2)
        return {
            Keys.VISUAL_FEAT: visual_feat.permute(2, 0, 1),
            Keys.CONV_LOGITS: None if logits is None else logits.permute(2, 0, 1),
            Keys.FEAT_LEN: lgt.cpu() if torch.is_tensor(lgt) else lgt,
        }


class sen_TemporalConv(nn.Module):
    """SEN 时间卷积模块的封装类。

    作为 SEN 模型中的 temporal_module_container 使用，
    从配置参数中解析设置并包装 SENTemporalConv。

    Args:
        args: 配置字典，包含 input_size, hidden_size, conv_type, num_classes 等
    """

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
        """前向传播。

        Args:
            data: 包含 Keys.FRAMEWISE_FEATURES 和 Keys.VID_LGT 的字典

        Returns:
            dict: 包含视觉特征、卷积 logits 和特征长度的字典
        """
        require(data, Keys.FRAMEWISE_FEATURES, Keys.VID_LGT, who="sen_TemporalConv")
        return self.conv1d(data[Keys.FRAMEWISE_FEATURES], data[Keys.VID_LGT])
