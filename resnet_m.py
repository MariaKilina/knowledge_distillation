import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import ResNet50_Weights

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, *args, **kwargs):
        super().__init__()
        self.in_channels, self.out_channels, self.stride = in_channels, out_channels, stride
        self.relu = nn.ReLU(inplace=True)
        self.expansion = expansion
        self.blocks = nn.Identity()
        if self.should_apply_downsampling:
            self.downsample = nn.Sequential(
                conv1x1(self.in_channels, self.expanded_channels, stride=stride),
                nn.BatchNorm2d(self.expanded_channels)
            )
        else:
            self.downsample = None        

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_downsampling(self):
        return self.in_channels != self.expanded_channels or self.stride != 1

    def forward(self, x):
        identity = x
        out = self.blocks(x)
        if self.should_apply_downsampling:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class BasicBlock(ResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, stride=stride, expantion=1, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

class BottleneckBlock(ResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, stride=stride, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv1x1(out_channels, self.expanded_channels),
            nn.BatchNorm2d(self.expanded_channels)
        )
        
def upsampleBlock(channel_in, channel_out, kernel_size=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1))
    )

def downsampleBlock(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )
"""
def shallowUpsample(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1))
    )

def shallowDownsample(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )
"""
        
class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes=100, width_decrement=1, *args, **kwargs):
        super().__init__()
        self.in_channels = 64//width_decrement
        self.width_decrement = width_decrement

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64//width_decrement, block, layers[0])
        self.layer2 = self._make_layer(128//width_decrement, block, layers[1], stride=2)
        self.layer3 = self._make_layer(256//width_decrement, block, layers[2], stride=2)
        self.layer4 = self._make_layer(512//width_decrement, block, layers[3], stride=2)

        self.fc = nn.Linear(self.in_channels*block.expansion, n_classes)

        self.upsample = upsampleBlock(self.in_channels, 512//width_decrement)
        
        self.downsample = downsampleBlock(512//width_decrement, self.in_channels)
        #self.upsample = shallowUpsample(self.in_channels, 512//width_decrement)
        #self.downsample = shallowDownsample(512//width_decrement, self.in_channels)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear((512 // width_decrement)*block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def _make_layer(self, out_channels, block_type, num_blocks, stride=1):
        layers = []
        if num_blocks == 0: return nn.Sequential(*layers)
        layers.append(block_type(self.in_channels, out_channels, stride=stride))
        self.in_channels = block_type.expansion * out_channels
        for _ in range(1, num_blocks):
            layers.append(block_type(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)
        x = self.downsample(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def resnet6(path='', pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [2, 0, 0, 0], **kwargs)
    if pretrained:
        model = model.load_state_dict(torch.load(path))
        model.eval()
    return model

def resnet10(path='', pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 0, 0], **kwargs)
    if pretrained:
        model = model.load_state_dict(torch.load(path))
        model.eval()
    return model
    
def resnet14(path='', pretrained=False, progress=True, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 0], **kwargs)
    if pretrained:
        model = model.load_state_dict(torch.load(path))
        model.eval()
    return model