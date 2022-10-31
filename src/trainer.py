from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings
from math import cos, pi
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
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len, cfg)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        acc.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg, acc.avg)

def validate(val_loader, val_loader_len, model, criterion, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        acc.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return (losses.avg, acc.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))
        
def adjust_learning_rate(optimizer, epoch, iteration, num_iter, cfg):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = cfg.warmup_epoch
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = cfg.epochs * num_iter

    if cfg.lr_decay == 'step':
        lr = cfg.lr * (cfg.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif cfg.lr_decay == 'cos':
        lr = cfg.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif cfg.lr_decay == 'linear':
        lr = cfg.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif cfg.lr_decay == 'schedule':
        count = sum([1 for s in cfg.schedule if s <= epoch])
        lr = cfg.lr * pow(cfg.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(cfg.lr_decay))

    if epoch < warmup_epoch:
        lr = cfg.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr