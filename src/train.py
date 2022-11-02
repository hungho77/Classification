from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings
import wandb
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from mobilenet import get_model
from utils import Logger, AverageMeter, accuracy, mkdir_p
from utils.dataloaders import *
from util import get_config
from trainer import *

best_prec = 0


def main(args):
    global best_prec
    # get config
    cfg = get_config(args.config)
    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cfg.distributed = cfg.world_size > 1

    if cfg.distributed:
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size)
        
    # init wandb log
    wandb.init(project=cfg.project,entity=cfg.entity,name=cfg.name) 
    wandb.config = {
        "learning_rate": cfg.lr,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "momentum": cfg.momentum,
        "weight_decay": cfg.weight_decay,
        "sample_rate": cfg.sample_rate
    }

    # create model
    print("=> Creating model '{}'".format(cfg.network))
    # Use pretrained imagenet
    if cfg.pretrained and not cfg.resume:
        if os.path.isfile(cfg.pretrained):
            print("=> loading pretrained '{}'".format(cfg.pretrained))
            model = get_model(cfg.network, num_classes=1000, width_mult=cfg.width_mult)
            state_dict = torch.load(cfg.pretrained)
            model.load_state_dict(state_dict)
            # Change output features of last linear layer
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, cfg.num_classes)
    # Do not use pretrained imagenet
    else:
        model = get_model(cfg.network, num_classes=cfg.num_classes, width_mult=cfg.width_mult)

    # Use distributed
    if not cfg.distributed:
        # model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    # optionally resume from a checkpoint
    title = "FaceMask_" + cfg.network
    if not os.path.isdir(cfg.output):
        mkdir_p(cfg.output)
    
    start_epoch = 0
        
    if cfg.resume:
        if os.path.isfile(cfg.checkpoint):
            print("=> loading checkpoint '{}'".format(cfg.checkpoint))
            checkpoint = torch.load(cfg.checkpoint)
            model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(cfg.checkpoint, checkpoint['epoch']))
            cfg.checkpoint = os.path.dirname(cfg.checkpoint)
            logger = Logger(os.path.join(cfg.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(cfg.checkpoint))
    else:
        logger = Logger(os.path.join(cfg.output, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    cudnn.benchmark = True

    # Data loading code
    if cfg.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    elif cfg.data_backend == 'dali-gpu':
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif cfg.data_backend == 'dali-cpu':
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()

    train_loader, train_loader_len = get_train_loader(cfg.data_dir, cfg.batch_size, workers=cfg.workers, input_size=cfg.input_size)
    
    val_loader, val_loader_len = get_val_loader(cfg.data_dir, cfg.batch_size, workers=cfg.workers, input_size=cfg.input_size)

    if cfg.evaluate:
        if os.path.isfile(cfg.weights):
            print("=> loading pretrained weight '{}'".format(cfg.weights))
            state_dict = torch.load(cfg.weights)
            model.load_state_dict(state_dict)
        else:
            print("=> no weight found at '{}'".format(cfg.weights))

        validate(val_loader, val_loader_len, model, criterion)
        return
    
    for epoch in range(start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion, optimizer, epoch, cfg)
        
        # evaluate on validation set
        val_loss, prec = validate(val_loader, val_loader_len, model, criterion, cfg)

        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec])
        
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        if is_best:
            save_checkpoint(model.state_dict(), 
                            is_best, checkpoint=cfg.output, 
                            filename='checkpoints_{}.pth'.format(epoch))
        
        # Wandb Logger
        log_dict = {"Learning Rate": lr, "Train Loss": train_loss, "Valid Loss": val_loss, "Train Acc" :train_acc, "Valid Acc": prec}
        wandb.log(log_dict)
        
        print('\nEpoch: [%d | %d], loss: %f, acc: %f' % (epoch + 1, cfg.epochs, train_loss, train_acc))
    
    logger.close()

    print('Best accuracy:')
    print(best_prec) 
    
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Face Mask Casification")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())