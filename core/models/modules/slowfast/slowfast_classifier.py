# modules/norm_linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.slowfast.TemporalSlowFastConv1D import TemporalSlowFastConv1D
from models.keys import Keys, require

"""SlowFast 分类器模块。

为 SlowFast 模型的多路径输出提供线性分类器，
支持权重归一化和分类器共享机制。
"""


class NormLinear(nn.Module):
    """权重归一化的线性层。

    使用 L2 归一化对权重矩阵进行归一化处理，
    然后执行矩阵乘法，替代标准的线性变换。

    Args:
        in_dim: 输入特征维度
        out_dim: 输出特征维度
    """

    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入张量，形状为 (..., in_dim)

        Returns:
            Tensor: 输出张量，形状为 (..., out_dim)
        """
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class slowfast_classifier(nn.Module):
    """SlowFast 分类器模块。

    为 SlowFast 的时序卷积输出提供三个路径（主路径、慢路径、快路径）
    的线性分类器，支持权重归一化和分类器共享机制。

    Args:
        args: 配置字典，包含 weight_norm, share_classifier, num_classes 等参数
    """

    def __init__(self, args):
        super(slowfast_classifier, self).__init__()
        weight_norm = args.get("weight_norm", None)
        share_classifier = args.get("share_classifier", None)
        self.num_classes = args.get("num_classes", None)
        self.conv1d = TemporalSlowFastConv1D(args)

        if weight_norm:
            self.classifier = nn.ModuleList([NormLinear(1024, self.num_classes) for i in range(3)])
            self.conv1d.fc = nn.ModuleList([NormLinear(1024, self.num_classes) for i in range(3)])
        else:
            self.classifier = nn.ModuleList([nn.Linear(1024, self.num_classes) for i in range(3)])
            self.conv1d.fc = nn.ModuleList([nn.Linear(1024, self.num_classes) for i in range(3)])

        if share_classifier == 1:
            self.conv1d.fc = self.classifier
        elif share_classifier == 2:
            classifier = self.classifier[0]
            self.classifier = nn.ModuleList([classifier for i in range(3)])
            self.conv1d.fc = nn.ModuleList([classifier for i in range(3)])

    def forward(self, data):
        """前向传播。

        对时序模型的输出 predictions 应用分类器，得到每个路径的序列 logits。

        Args:
            data: 包含 Keys.PREDICTIONS 的字典

        Returns:
            dict: 包含 Keys.SEQUENCE_LOGITS 的字典
        """
        require(data, Keys.PREDICTIONS, who="slowfast_classifier")

        predictions = data[Keys.PREDICTIONS]

        # 如果 predictions 是列表, 对每个预测执行分类器
        if isinstance(predictions, list):
            sequence_logits = [self.classifier[i](pred) for i, pred in enumerate(predictions)]
        else:
            # 如果 predictions 不是列表, 直接使用第一个分类器
            sequence_logits = self.classifier[0](predictions)

        return {
            Keys.SEQUENCE_LOGITS: sequence_logits
        }
