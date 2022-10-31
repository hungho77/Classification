from __future__ import absolute_import
from .mobilenetv1 import *
from .mobilenetv2 import *
from .mobilenetv3 import *

def get_model(name, **kwargs):
    # mobilenet
    if name == "mobilenetv1":
        return mobilenetv1(**kwargs)
    elif name == "mobilenetv2":
        return mobilenetv2(**kwargs)
    elif name == "mobilenetv3_large":
        return mobilenetv3_large(**kwargs)
    elif name == "mobilenetv3_small":
        return mobilenetv3_small(**kwargs)
    else:
        raise ValueError()