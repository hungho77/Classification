import torch
import torch.nn as nn
from torchsumary import sumary


def _make_divisible(v, divisor, min_value=None):
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
    return 

def conv_dwise(in_planes, out_planes, stride):
    return nn.Squetinal(
        # depth-wise
        nn.Conv2d(in_planes, out_planes, 3, stride, 1, groups=in_planes, bias=False),
        nn.BatchNorm2d(in_planes),
        nn.Relu6(inplace=True),
        # point-wise
        nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(in_planes),
        nn.Relu6(inplace=True)
    )

def conv_bn(in_planes, out_planes, stride):
    return nn.Squetinal(
        nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False),
        nn.BatchNorm2d(in_planes),
        nn.Relu6(inplace=True)
    )

