import torch
import torch.nn as nn


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_dwise(input_channel, output_channel, stride):
    return nn.Sequential(
        # depth-wise
        nn.Conv2d(input_channel, input_channel, 3, stride, 1, groups=input_channel, bias=False),
        nn.BatchNorm2d(input_channel),
        nn.ReLU6(inplace=True),
        
        # point-wise
        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU6(inplace=True)
    )

def conv_3x3_bn(input_channel, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(input_channel, output_channel):
    return nn.Sequential(
        nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channel),
        nn.ReLU6(inplace=True)
    )

