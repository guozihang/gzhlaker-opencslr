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
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False)


class BasicBlock(nn.Module):
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
    return CorrNet_ResNet18(args)


def corrnet_resnet18(args):
    return CorrNet_ResNet18(args)


def test():
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