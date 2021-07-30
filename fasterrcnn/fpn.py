"""
@author: tangjun
@contact: 511026664@qq.com
@time: 2021/06/29 22:34
@desc: fpn代码编写
"""

import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],  # 输入的各个stage的通道数,用来匹配backbone
                 out_channels=256,  # 输出的特征层的通道数
                 num_outs=5,  # fpn返回特征金字塔的层数
                 add_extra_convs=False,  # 取值为[False on_input on_lateral on_output]
                 relu_before_extra_convs=False,
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.add_extra_convs = add_extra_convs
        # Module 和 list 的结合
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(self.in_channels)):  # 起始与结束层
            l_conv = self.Conv(
                in_channels[i],  # [256, 512, 1024, 2048]
                out_channels,  # 256
                kernel_size=1,
                padding=0,
                bias=True
            )  # 采用1*1输出每一层通道为out_channels (256)
            fpn_conv = self.Conv(
                out_channels,
                out_channels,  # 256
                kernel_size=3,  # kernel
                padding=1,
                bias=True
            )  # 采用3*3提取特征

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = self.num_outs - len(self.in_channels)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = self.Conv(
                    in_channels,
                    out_channels,  # 256
                    kernel_size=3,  # kernel
                    stride=2,
                    padding=1,
                    bias=True
                )  # 采用3*3提取特征

                self.fpn_convs.append(extra_fpn_conv)

    def Conv(self, in_planes, out_planes, **kwargs):
        "3x3 convolution with padding"
        padding = kwargs.get('padding', 1)
        bias = kwargs.get('bias', False)
        stride = kwargs.get('stride', 1)
        kernel_size = kwargs.get('kernel_size', 3)
        out = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels), 'stages of inputs from backboe not match fpn stage in fpn.py'
        # build laterals 将C1-C4 inputs通道转为统一通道256
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]  # 表示输入函数的值
        # build top-down path 上采样特征融合
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode='nearest')
        #                        上采样函数    输入值      原图尺寸倍数     采样方法

        # build outputs
        # part 1: from original levels  P2-P4 特征提取
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        # part 2: add extra levels eg:P6
        if self.num_outs > len(outs):
            if self.add_extra_convs:  # add conv layers on top of original feature maps (RetinaNet)
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError(
                        'extra_source must be on_input on_lateral or on_output,but val of {}'.format(
                            self.add_extra_convs))
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):  # 输出其它层方法，一般不用
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
            else:  # # use max pool to get more levels on top of outputs (e.g., Faster R-CNN, Mask R-CNN)
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)


if __name__ == '__main__':
    import torch

    C = []
    in_channels = [256, 512, 1024, 2048]
    for j, i in enumerate([2, 4, 8, 16]):
        V = tuple([3, in_channels[j], int(128 / i), int(128 / i)])
        layer = torch.ones(V) * 255.0
        C.append(layer)

    C = tuple(C)
    model = FPN()
    out = model(C)
    print(out)
