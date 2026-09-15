"""提升池化(LiftPool)与局部加权。

从 temporal/tconv.py 里搬出来的辅助积木:TemporalLiftPooling 是
tconv.TemporalConv 在 kernel_size 写 'P<n>' 时用的池化层(唯一带可学习参数的
池化,顺带产出 Cu/Cp 两项辅助损失),Local_Weighting 只被它自己用。
两者都不是完整的时序网络,所以按「辅助积木进 others/」的规矩放这里。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLiftPooling(nn.Module):
    """时序提升池化（LiftPool）模块。

    基于提升方案（Lifting Scheme）的时序池化操作，通过预测器（Predictor）
    和更新器（Updater）自适应地分解和池化时序特征。

    Args:
        input_size: 输入特征通道数。
        kernel_size: 池化核大小（间隔），默认为2。
    """

    def __init__(self, input_size, kernel_size=2):
        super(TemporalLiftPooling, self).__init__()
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
