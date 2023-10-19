'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from mup import MuReadout

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        layers = [self.conv1, self.conv2]
        if len(self.shortcut) > 1:
            layers.append(self.shortcut[0])
        for layer in layers:
            init.kaiming_normal_(layer.weight, a=1)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
            self.reset_parameters()

    def reset_parameters(self) -> None:
        layers = [self.conv1, self.conv2, self.conv3]
        if len(self.shortcut) > 1:
            layers.append(self.shortcut[0])
        for layer in layers:
            init.kaiming_normal_(layer.weight, a=1)
            if layer.bias is not None:
                init.zeros_(layer.bias)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNet, self).__init__()

        base_widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in base_widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        ### This is the only μP related change ###
        self.linear = MuReadout(feat_scale*widths[3]*block.expansion, num_classes, readout_zero_init=True)
        ###########################################

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)

        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        return final

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def ResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=5, **kwargs)

def ResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wm=.75, **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)

resnet50 = ResNet50
resnet18 = ResNet18
resnet101 = ResNet101
resnet152 = ResNet152
resnet18wide = ResNet18Wide