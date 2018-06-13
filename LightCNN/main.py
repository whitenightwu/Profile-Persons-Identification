'''
    implement training process for Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''
from __future__ import print_function

import argparse
import os
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from light_cnn import LightCNN_9Layers, LightCNN_29Layers_v2
from load_imglist import ImageList
from train import train, validate

parser = argparse.ArgumentParser(description='PyTorch Light CNN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')

parser.add_argument('--model', default='LightCNN-9', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29, LightCNN-29v2')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--root_path', default='', type=str, metavar='PATH',
                    help='path to images (default: none)')
parser.add_argument('--train_list', default='', type=str, metavar='PATH',
                    help='path to training list (default: none)')
parser.add_argument('--val_list', default='', type=str, metavar='PATH',
                    help='path to validation list (default: none)')
parser.add_argument('--save_path', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--num_classes', default=4382, type=int,
                    metavar='N', help='number of classes (default: 4382)')
parser.add_argument('--end2end', action='store_true',
                    help='if true, using end2end with dream block, else, using naive architecture')

def main():
    global args
    args = parser.parse_args()

    model = create_model(args.end2end)
    params = create_model_parameters(args, model)

    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()   # loss function

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # load image
    train_loader = load_image(args.train_list,
                              transforms.Compose(
                                [transforms.RandomCrop(128),
                                    transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), ])
                              , True, True)

    val_loader = load_image(args.val_list,
                            transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor(), ]),
                            False, True)

    if args.cuda:
        criterion.cuda()

    validate(val_loader, model, criterion)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        save_name = args.save_path + 'lightCNN_' + str(epoch + 1) + '_checkpoint.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'prec1': prec1,
        }, save_name)


def load_image(fileList, transforms, shuffle=True, pin_memory=True):
    loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=fileList, transform=transforms),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=pin_memory)
    return loader


def create_model_parameters(args, model):
    params = []
    for name, value in model.named_parameters():
        if 'bias' in name:
            if 'fc2' in name:
                params += [{'params': value, 'lr': 20 * args.lr, 'weight_decay': 0}]
            else:
                params += [{'params': value, 'lr': 2 * args.lr, 'weight_decay': 0}]
        else:
            if 'fc2' in name:
                params += [{'params': value, 'lr': 10 * args.lr}]
            else:
                params += [{'params': value, 'lr': 1 * args.lr}]
    return params


def create_model(end2end=True):
    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(end2end, num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(end2end, num_classes=args.num_classes)
    else:
        print('Error model type\n')
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    # print(model)
    return model


def save_checkpoint(state, filename):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch):
    scale = 0.457305051927326
    step = 10
    lr = args.lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


if __name__ == '__main__':
    main()
