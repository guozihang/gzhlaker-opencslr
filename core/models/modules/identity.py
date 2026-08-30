"""恒等映射模块。

提供恒等映射层，用于替换预训练网络的分类头以保留特征向量。
"""

import torch.nn as nn


class Identity ( nn.Module ) :
    """恒等映射层。

    直接返回输入，不进行任何变换。
    用于替换 torchvision 预训练模型的 fc 层，以保留特征向量而非分类结果。
    """

    def __init__ ( self ) :
        super ( Identity , self ).__init__ ( )

    def forward ( self , x ) :
        """前向传播（恒等映射）。

        Args:
            x: 输入张量

        Returns:
            torch.Tensor: 未经修改的输入张量
        """
        return x