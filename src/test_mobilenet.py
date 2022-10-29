import torchvision.models as models

import mobilenet

default_mobilenet = sorted(name for name in mobilenet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(mobilenet.__dict__[name]))

customized_mobilenet_version = sorted(name for name in mobilenet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(mobilenet.__dict__[name]))

for name in mobilenet.__dict__:
    if name.islower() and not name.startswith("__") and callable(mobilenet.__dict__[name]):
        models.__dict__[name] = mobilenet.__dict__[name]

mobilenet_version = default_mobilenet + customized_mobilenet_version

# create model
print("=> creating model '{}'".format(args.arch))
model = models.__dict__[args.arch](width_mult=args.width_mult)

if not args.distributed:
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
else:
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)