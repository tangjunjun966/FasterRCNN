"""
@author: tangjun
@contact: 511026664@qq.com
@time: 2020/12/7 22:48
@desc: 残差ackbone改写，用于构建特征提取模块
"""

import torch.nn as nn
import torch
from collections import OrderedDict


def Conv(in_planes, out_planes, **kwargs):
    "3x3 convolution with padding"
    padding = kwargs.get('padding', 1)
    bias = kwargs.get('bias', False)
    stride = kwargs.get('stride', 1)
    kernel_size = kwargs.get('kernel_size', 3)
    out = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth=50,
                 in_channels=None,
                 pretrained=None,
                 frozen_stages=-1
                 # num_classes=None
                 ):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.inchannels = in_channels if in_channels is not None else 3  # 输入通道
        # self.num_classes=num_classes
        self.block, layers = self.arch_settings[depth]
        self.frozen_stages = frozen_stages
        self.conv1 = nn.Conv2d(self.inchannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2)

        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * self.block.expansion, self.num_classes)
        self._freeze_stages()  # 冻结函数
        if pretrained is not None:
            self.init_weights(pretrained=pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.load_checkpoint(pretrained)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='relu')
                    if hasattr(m, 'bias') and m.bias is not None:  # m包含该属性且m.bias非None # hasattr(对象，属性)表示对象是否包含该属性
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def load_checkpoint(self, pretrained):

        checkpoint = torch.load(pretrained)
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']

        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}

        unexpected_keys = []  # 保存checkpoint不在module中的key
        model_state = self.state_dict()  # 模型变量

        for name, param in state_dict.items():  # 循环遍历pretrained的权重
            if name not in model_state:
                unexpected_keys.append(name)
                continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data

            try:
                model_state[name].copy_(param)  # 试图赋值给模型
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, '
                    'whose dimensions in the model are {} not equal '
                    'whose dimensions in the checkpoint are {}.'.format(
                        name, model_state[name].size(), param.size()))
        missing_keys = set(model_state.keys()) - set(state_dict.keys())
        print('missing_keys:', missing_keys)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return tuple(outs)


if __name__ == '__main__':
    x = torch.ones((2, 3, 215, 215))
    model = Resnet(depth=50)

    model.init_weights(pretrained='./resnet50.pth')

    out = model(x)

    print(out)
