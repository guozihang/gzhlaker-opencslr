"""CorrNet 增强的 3D ResNet 骨干网络，用于空间特征提取。

在标准 ResNet 的层间插入相关性建模模块（Correlation Module），
通过帧间相关性建模增强视频特征表示，适用于连续手语识别。
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from models.keys import Keys, require

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200', 'CorrNet_ResNet18', 'corrnet_resnet18'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Get_Correlation(nn.Module):
    """相关性计算模块（Correlation Module）。

    通过帧间注意力机制计算视频帧之间的相关性，并结合多尺度空间聚合
    的识别模块，增强特征表示。

    Args:
        channels: 输入特征通道数。
    """

    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels // 16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 1, 1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 2, 2), dilation=(1, 2, 2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9, 3, 3),
                                              padding=(4, 3, 3), dilation=(1, 3, 3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):
        """前向传播。

        计算帧间相关性和多尺度空间聚合特征，输出增强的特征。

        Args:
            x: 输入特征，形状 (B, C, T, H, W)

        Returns:
            torch.Tensor: 相关性增强的特征，形状与输入相同
        """
        # 相关性计算 (Correlation Module)
        x2 = self.down_conv2(x)
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2))
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2))
        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, 1:], x2[:, :, -1:]], 2),
                                F.sigmoid(affinities) - 0.5) * self.weights2[0] + \
                   torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:, :, :1], x2[:, :, :-1]], 2),
                                F.sigmoid(affinities2) - 0.5) * self.weights2[1]

        # 识别模块 (Identification Module) - 多尺度空间聚合
        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x) * self.weights[0] + \
                       self.spatial_aggregation2(x) * self.weights[1] + \
                       self.spatial_aggregation3(x) * self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        # 结合相关性和识别结果
        return features * (F.sigmoid(aggregated_x) - 0.5)


def conv3x3(in_planes, out_planes, stride=1):
    """创建 3x3 的 3D 卷积层（时序维度为 1）。

    Args:
        in_planes: 输入通道数。
        out_planes: 输出通道数。
        stride: 空间维度的步长。

    Returns:
        nn.Conv3d: 配置好的卷积层。
    """
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False)


class BasicBlock(nn.Module):
    """ResNet 基本残差块（3D 版本）。

    由两个 3x3 卷积层组成，包含批归一化和 ReLU 激活，
    通过残差连接缓解梯度消失问题。

    Args:
        inplanes: 输入通道数。
        planes: 中间和输出通道数。
        stride: 卷积步长。
        downsample: 可选的下采样层，用于匹配残差连接的维度。
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """前向传播。

        Args:
            x: 输入特征，形状 (B, C, T, H, W)

        Returns:
            torch.Tensor: 残差块输出特征，形状与输入相同（通道数可能变化）
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """CorrNet 增强的 3D ResNet 网络。

    在标准 ResNet 的 layer2、layer3、layer4 之后分别插入
    Get_Correlation 模块，通过可学习的 alpha 参数控制相关性
    增强的强度，初始为 0（无增强效果）。

    Args:
        block: 残差块类型（如 BasicBlock）。
        layers: 每个阶段的残差块数量列表。
        num_classes: 分类数（默认 1000）。
    """

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.corr1 = Get_Correlation(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr2 = Get_Correlation(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr3 = Get_Correlation(self.inplanes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建 ResNet 的一个阶段。

        由多个残差块组成，可选择下采样以匹配通道数和空间尺寸。

        Args:
            block: 残差块类型。
            planes: 输出通道数。
            blocks: 残差块数量。
            stride: 第一个残差块的卷积步长。

        Returns:
            nn.Sequential: 包含多个残差块的顺序容器。
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播。

        依次通过卷积层、池化层和四个残差阶段，
        在 layer2、layer3、layer4 后插入相关性增强，
        最后通过全局平均池化和全连接层输出分类结果。

        Args:
            x: 输入视频，形状 (N, C, T, H, W)

        Returns:
            torch.Tensor: 分类输出，形状 (N, num_classes)
        """
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = x + self.corr1(x) * self.alpha[0]  # 残差连接，alpha初始为0
        x = self.layer3(x)
        x = x + self.corr2(x) * self.alpha[1]
        x = self.layer4(x)
        x = x + self.corr3(x) * self.alpha[2]

        # 转换为帧级特征
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])  # bt,c,h,w
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # bt,c
        x = self.fc(x)  # bt,num_classes

        return x


def resnet18(**kwargs):
    """构建 ResNet-18 模型并加载预训练权重。

    加载 2D ResNet-18 的 ImageNet 预训练权重，将卷积层权重扩展为 3D，
    过滤掉分类层权重以保持特征提取能力。

    Args:
        **kwargs: 传递给 ResNet 构造函数的额外参数。

    Returns:
        ResNet: 加载了预训练权重的 ResNet-18 模型。
    """
    """Constructs a ResNet-18 based model."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())

    # 过滤掉分类层权重
    filtered_checkpoint = {}
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
        # 只保留特征提取层的权重，跳过fc层
        if not ln.startswith('fc.'):
            filtered_checkpoint[ln] = checkpoint[ln]

    model.load_state_dict(filtered_checkpoint, strict=False)
    return model


class CorrNet_ResNet18(nn.Module):
    """CorrNet 增强的 ResNet18 空间特征提取模块。

    使用 CorrNet 增强的 ResNet18 提取帧级视觉特征。
    移除原始 ResNet 的分类层和全局池化，替换为自适应池化以输出帧级特征。
    按顺序执行每层前向传播，在 layer2、layer3、layer4 之后插入相关性增强。

    Args:
        args: 配置字典，可包含 "num_classes"（预训练所需，默认 1000）。
    """

    def __init__(self, args):
        super(CorrNet_ResNet18, self).__init__()
        # 加载原始CorrNet ResNet18
        self.resnet = resnet18(num_classes=args.get("num_classes", 1000))

        # 手动移除分类层和全局池化层
        if hasattr(self.resnet, 'fc'):
            del self.resnet.fc
        if hasattr(self.resnet, 'avgpool'):
            del self.resnet.avgpool

        # 不再使用 Sequential，而是直接使用 resnet 的各个层
        # 这样我们可以控制前向传播过程，包括 CorrNet 模块
        self.feature_dim = 512
        self.frame_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, data):
        """前向传播。

        从数据字典中读取视频，执行 CorrNet 增强的 ResNet18 前向传播，
        输出帧级特征。

        Args:
            data: 数据字典，需包含:
                Keys.VID: 视频帧，形状 (B, T, C, H, W)
                Keys.VID_LGT: 每个视频的实际帧数（可选）

        Returns:
            dict: 包含 Keys.FRAMEWISE_FEATURES 和 Keys.VID_LGT 的字典
        """
        require(data, Keys.VID, who="CorrNet_ResNet18")

        vid = data[Keys.VID]  # 形状: (B, T, C, H, W)

        # 转换维度: (B, T, C, H, W) -> (B, C, T, H, W)
        frames = vid.permute(0, 2, 1, 3, 4).contiguous()

        B, C, T, H, W = frames.shape

        # 确保通道数是3
        if C != 3:
            raise ValueError(f"输入通道数应为3，但得到: {C}")

        # 手动执行前向传播，包含 CorrNet 模块
        x = frames

        # 第一层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # layer1
        x = self.resnet.layer1(x)

        # layer2 + corr1
        x = self.resnet.layer2(x)
        x = x + self.resnet.corr1(x) * self.resnet.alpha[0]

        # layer3 + corr2
        x = self.resnet.layer3(x)
        x = x + self.resnet.corr2(x) * self.resnet.alpha[1]

        # layer4 + corr3
        x = self.resnet.layer4(x)
        x = x + self.resnet.corr3(x) * self.resnet.alpha[2]

        # 全局池化得到帧级特征
        frame_features = self.frame_pool(x)  # (B, 512, T, 1, 1)
        frame_features = frame_features.squeeze(-1).squeeze(-1)  # (B, 512, T)

        return {
            Keys.FRAMEWISE_FEATURES: frame_features,
            Keys.VID_LGT: data.get(Keys.VID_LGT, torch.tensor([T] * B, device=frames.device)),
        }


def corrnet_resnet18(args):
    """CorrNet_ResNet18 的工厂函数。

    Args:
        args: 配置字典。

    Returns:
        CorrNet_ResNet18: 构建好的模型实例。
    """
    return CorrNet_ResNet18(args)


def test():
    """测试 CorrNet_ResNet18 的前向传播。

    创建模型实例，输入随机视频数据，验证输出特征形状正确。
    """
    # 测试CorrNet版本的ResNet
    net = CorrNet_ResNet18({})
    test_data = {
        'vid': torch.randn(2, 16, 224, 224, 3),  # (B, T, H, W, C)
        'vid_lgt': torch.tensor([16, 16])
    }
    output = net(test_data)
    print("输出特征形状:", output['framewise_features'].shape)
    # 应该输出: torch.Size([2, 512, 16])


if __name__ == "__main__":
    test()
