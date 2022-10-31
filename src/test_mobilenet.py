import torch
import torchvision.models as models

from mobilenet import get_model
from mobilenet import mobilenetv1


# create model
model_names = "mobilenetv2"
print("=> creating model '{}'".format(model_names))
model = get_model(model_names, num_classes=2, width_mult=1.0)
input = torch.rand((1, 3, 224, 224))
out = model(input)
print(out.shape)
