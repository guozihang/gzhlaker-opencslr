"""归一化线性层模块。

提供权重归一化的线性层，用于替代标准 nn.Linear。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormLinear(nn.Module):
    """权重归一化的线性层。

    对权重矩阵进行 L2 归一化后执行矩阵乘法，有助于稳定训练。

    Args:
        in_dim: 输入特征维度。
        out_dim: 输出特征维度。
    """

    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入特征，形状 (..., in_dim)

        Returns:
            torch.Tensor: 输出特征，形状 (..., out_dim)
        """
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs
