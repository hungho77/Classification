import torch.nn as nn
from torchsummary import summary
import math

from net import *

__all__ = ['mobilenetv2']


class MobileNetV1(nn.Module):
    def __init__(self, in_planes, n_classes, width_mult=1.):
        super(MobileNetV1, self).__init__()

        # setting of depth-wise blocks
        self.cfgs = [
            # c,  s
            [ 64, 1],
            [128, 2],
            [128, 1],
            [256, 2],
            [256, 1],
            [512, 2],
            [512, 1],
            [512, 1],
            [512, 1],
            [512, 1],
            [512, 1],
            [1024, 2],
            [1024, 2]
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building depth-wise blocks
        for c, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            layers.append(conv_dwise(input_channel, output_channel, s if i == 0 else 1))
            input_channel = output_channel
        self.feature = nn.Sequential(*layers)
        
        # building last several layers
        output_channel = _make_divisible(1024 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1024
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def foward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
def mobilenetv1(**kwargs):
    """
    Constructs a MobileNet V1 model
    """
    return MobileNetV1(**kwargs)

