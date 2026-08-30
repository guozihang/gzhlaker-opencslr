"""SEN-based ResNet backbone for sign language recognition.

Implements a ResNet-18 variant enhanced with Temporal-Spatial Enhancement Module (TSEM)
and Spatial Enhancement Module (SSEM) for video-based feature extraction.
"""
import torch
from torch import nn
import torch
import torch.nn.functional as F
from wandb.util import downsample
from whisper import aggregate
import torch.utils.model_zoo as model_zoo
from ..modules.identity import Identity
from ..keys import Keys, require

model_urls={
    #'resnet18':'https://download.pytorch.org/models/resnet18-f37072fd.pth'
    'resnet18':'mymodels/resnet18-f37072fd.pth'
}
def conv3x3(in_planes,out_planes,stride=1):
    """Create a 3x3 3D convolution layer.

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        stride: Convolution stride along spatial dimensions.

    Returns:
        nn.Conv3d: A 3D convolutional layer with kernel size (1,3,3).
    """
    return nn.Conv3d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False
    )

class TSEM(nn.Module):
    """Temporal-Spatial Enhancement Module (TSEM).

    Enhances temporal features using multi-scale dilated convolutions
    with learnable weights and a gating mechanism.
    """

    def __init__(self,input_size):
        super(TSEM,self).__init__()
        hidden_size=input_size//16
        self.conv_transform=nn.Conv1d(input_size,hidden_size,kernel_size=1,stride=1,padding=0)
        self.conv_back=nn.Conv1d(hidden_size,input_size,kernel_size=1,stride=1,padding=0)
        self.num=5
        self.conv_enhance=nn.ModuleList([
            nn.Conv1d(hidden_size,hidden_size,kernel_size=3,stride=1,padding=int(i+1),groups=hidden_size,dilation=int(i+1)) for i in range(self.num)
        ])
        self.weights=nn.Parameter(torch.ones(self.num)/self.num,requires_grad=True)
        self.alpha=nn.Parameter(torch.zeros(1),requires_grad=True)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        """Forward pass of TSEM.

        Args:
            x: Input tensor of shape (N, C, T, H, W).

        Returns:
            Tensor: Enhanced feature map with the same shape as input.
        """
        out=self.conv_transform(x.mean(-1).mean(-1))
        aggregate_out=0
        for i in range(self.num):
            aggregate_out+=self.conv_enhance[i](out)*self.weights[i]
        out=self.conv_back(aggregate_out)
        return x*(torch.sigmoid(out.unsqueeze(-1).unsqueeze(-1))-0.5)*self.alpha

class SSEM(nn.Module):
    """Spatial Enhancement Module (SSEM).

    Enhances spatial features using multi-scale 3D dilated convolutions
    with learnable weights and a gating mechanism.
    """

    def __init__(self,input_size):
        super(SSEM,self).__init__()
        div_channel=input_size//16
        self.conv_transform=nn.Conv3d(input_size,div_channel,kernel_size=(1,1,1))
        self.num=3
        self.conv_enhance=nn.ModuleList([
            nn.Conv3d(div_channel,div_channel,kernel_size=(9,3,3),padding=(4,i+1,i+1),dilation=(1,i+1,i+1),groups=div_channel) for i in range(self.num)
        ])
        self.weights=nn.Parameter(torch.ones(self.num)/self.num,requires_grad=True)
        self.conv_back=nn.Conv3d(div_channel,input_size,kernel_size=(1,1,1))
        self.alpha=nn.Parameter(torch.ones(1),requires_grad=True)
    def forward(self,x):
        """Forward pass of SSEM.

        Args:
            x: Input tensor of shape (N, C, T, H, W).

        Returns:
            Tensor: Enhanced feature map with the same shape as input.
        """
        out=self.conv_transform(x)
        aggregate_out=0
        for i in range(self.num):
            aggregate_out+=self.conv_enhance[i](out)*self.weights[i]
        out=self.conv_back(aggregate_out)
        return x*(torch.sigmoid(out)-0.5)*self.alpha

class BasicBlock(nn.Module):
    """Basic residual block with TSEM and SSEM enhancements.

    A 3D residual block that incorporates temporal-spatial and spatial
    enhancement modules before the second convolution.

    Attributes:
        expansion: Expansion factor for output channels.
    """
    expansion=1
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm3d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm3d(planes)
        self.downsample=downsample
        self.tsem=TSEM(planes)
        self.ssem=SSEM(planes)
    def forward(self,x):
        """Forward pass of BasicBlock.

        Args:
            x: Input tensor of shape (N, C, T, H, W).

        Returns:
            Tensor: Output tensor after residual connection and activation.
        """
        residual=x
        #print('x.shape:',x.shape)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=out+self.ssem(out)+self.tsem(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        return out
class ResNet(nn.Module):
    """ResNet backbone with 3D convolutions for video processing.

    Implements a standard ResNet architecture using 3D convolutions
    for spatio-temporal feature extraction.

    Args:
        block: The residual block type (e.g., BasicBlock).
        layers: Number of blocks in each layer.
        num_classes: Number of output classes for classification.
    """

    def __init__(self,block,layers,num_classes=1000):
        self.inplanes=64
        super(ResNet,self).__init__()
        self.conv1=nn.Conv3d(3,64,kernel_size=(1,7,7),stride=(1,2,2),padding=(0,3,3),bias=False)
        self.bn1=nn.BatchNorm3d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool3d(kernel_size=(1,3,3),stride=(1,2,2),padding=(0,1,1))
        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)
        self.avgpool=nn.AvgPool2d(7,stride=1)
        self.fc=nn.Linear(512*block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm3d) or isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def _make_layer(self,block,planes,blocks,stride=1):
        """Create a sequential layer of residual blocks.

        Args:
            block: The residual block type.
            planes: Number of output channels for the layer.
            blocks: Number of blocks in the layer.
            stride: Convolution stride for the first block.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(
                nn.Conv3d(self.inplanes,planes*block.expansion,kernel_size=1,stride=(1,stride,stride),bias=False),
                nn.BatchNorm3d(planes*block.expansion),
            )
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
    def forward(self,x):
        """Forward pass of ResNet.

        Args:
            x: Input tensor of shape (N, C, T, H, W).

        Returns:
            Tensor: Output feature vector after global pooling and FC.
        """
        #print(x.shape)
        #N,C,T,H,W=x.size()
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=x.transpose(1,2).contiguous()
        x=x.view((-1,)+x.size()[2:])
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return  x
def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model

class SENresnet(nn.Module):
    """SEN-ResNet wrapper for sign language recognition.

    Wraps a ResNet-18 backbone and processes video frames to extract
    frame-wise visual features. Handles both 5D video tensors and
    pre-extracted features.

    Args:
        args: Configuration dictionary containing model parameters.
    """

    def __init__(self,args):
        super(SENresnet,self).__init__()
        self.conv2d=resnet18()
        self.conv2d.fc=Identity()
    def masked_bn ( self , inputs , len_x ) :
        """Apply masked batch normalization with variable-length inputs.

        Args:
            inputs: Input tensor.
            len_x: List of lengths for each sample in the batch.

        Returns:
            Tensor: Output after batch normalization, padded to uniform length.
        """
        def pad ( tensor , length ) :
            return torch.cat (
                [ tensor , tensor.new ( length - tensor.size ( 0 ) , *tensor.size ( ) [ 1 : ] ).zero_ ( ) ] )

        x = torch.cat ( [ inputs [ len_x [ 0 ] * idx :len_x [ 0 ] * idx + lgt ] for idx , lgt in enumerate ( len_x ) ] )
        x = self.conv2d ( x )
        x = torch.cat ( [ pad ( x [ sum ( len_x [ :idx ] ) :sum ( len_x [ :idx + 1 ] ) ] , len_x [ 0 ] )
                          for idx , lgt in enumerate ( len_x ) ] )
        return x
    def forward(self,data):
        """Forward pass of SEN-ResNet.

        Extracts frame-wise features from video input. Supports both
        raw video tensors (5D) and pre-extracted features.

        Args:
            data: Dict containing Keys.VID with video tensor
                  of shape (N, C, T, H, W) or (N, T, D).

        Returns:
            dict: Dict containing Keys.FRAMEWISE_FEATURES.
        """
        require(data, Keys.VID, who="SENresnet")
        x=data[Keys.VID]
        if len(x.shape)==5:
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0,2,1,3,4))
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            #print(framewise.shape)
        else:
            framewise=x
        return {
            Keys.FRAMEWISE_FEATURES: framewise,
        }



























































