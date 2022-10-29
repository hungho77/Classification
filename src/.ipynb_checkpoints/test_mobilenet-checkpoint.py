import torch
import torchvision.models as models

import mobilenet

default_mobilenet_version = sorted(name for name in mobilenet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(mobilenet.__dict__[name]))

customized_mobilenet_version = sorted(name for name in mobilenet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(mobilenet.__dict__[name]))

for name in mobilenet.__dict__:
    if name.islower() and not name.startswith("__") and callable(mobilenet.__dict__[name]):
        models.__dict__[name] = mobilenet.__dict__[name]

mobilenet_version = default_mobilenet_version + customized_mobilenet_version

# create model
model_names = "mobilenetv1"
print("=> creating model '{}'".format(model_names))
model = models.__dict__[model_names](width_mult=1.0)
input = torch.rand((1, 3, 224, 224))
out = model(input)
print(out.shape)
# if not args.distributed:
#     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
#         model.features = torch.nn.DataParallel(model.features)
#         model.cuda()
#     else:
#         model = torch.nn.DataParallel(model).cuda()
# else:
#     model.cuda()
#     model = torch.nn.parallel.DistributedDataParallel(model)