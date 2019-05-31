#!/usr/bin/env python
# coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import sys
import time
import numpy as np

from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
sys.path.append('../models')
from models.densenet import densenet121, densenet161,densenet169,densenet201
from models.resnext import resnext101, resnext152
from models.senet import se_resnet152,se_resnext101_32x4d
from models.inceptionresnetv2 import  inceptionresnetv2
from models.inceptionv4 import inceptionv4
from models.dpn import dpn131

import dataloader
import parser
args = parser.args
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

def get_lr(epoch):
    if epoch <= args.epochs * 0.5:
        lr = args.lr
    elif epoch <= args.epochs :
        lr = 0.1 * args.lr
    return lr


def train(data_loader,net,loss,epoch,optimizer):
    net.train ( )
    lr = get_lr ( epoch )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss_list = []
    accuracy_list = []
    for b, (data, visit_array,label, name) in enumerate ( tqdm ( data_loader ) ):
        data = data.type ( torch.FloatTensor )
        data = Variable ( data.cuda ( async=True ) )
        label = label.type ( torch.LongTensor )
        label = Variable ( label.cuda ( async=True ) )
        output = net ( data )
        loss_output = loss ( output, label )
        loss_list.append ( loss_output.data.cpu ( ).numpy ( ) )
        optimizer.zero_grad ( )
        loss_output.backward ( )
        optimizer.step ( )

        pred = np.argmax ( output.data.cpu ( ).numpy ( ), axis=1 )
        label = label.data.cpu ( ).numpy ( )
        correct = sum(pred == label)
        accuracy = correct*1.0 / pred.shape[0]
        accuracy_list.append(accuracy)
    accuracy = np.mean(accuracy_list)
    print('Train Epoch %03d (lr %.5f)' % (epoch, lr))
    print 'train loss: {:3.4f} \t'.format ( np.mean ( loss_list ) )
    print 'train accuracy: {:3.4f}'.format ( accuracy )
    print


def test(data_loader, net, loss, epoch, best_acurracy,phase='valid'):
    net.eval ( )
    loss_list = []
    accuracy_list = []

    for b, (data,visit_array, label, names) in enumerate ( tqdm ( data_loader ) ):
        data = data.type ( torch.FloatTensor )
        data = Variable ( data.cuda ( async=True ) )

        label = label.type ( torch.LongTensor )
        label = Variable ( label.cuda ( async=True ) )
        output = net ( data )
        loss_output = loss ( output, label )
        loss_list.append ( loss_output.data.cpu ( ).numpy ( ) )

        pred = np.argmax ( output.data.cpu ( ).numpy ( ), axis=1 )
        label = label.data.cpu ( ).numpy ( )
        correct = sum ( pred == label )
        accuracy = correct * 1.0 / pred.shape[0]
        accuracy_list.append ( accuracy )
    accuracy = np.mean ( accuracy_list )
    print('%s Epoch %03d ' % (phase, epoch))
    print 'valid loss: {:3.4f} \t'.format ( np.mean ( loss_list ) )
    print 'valid accuracy: {:3.4f}'.format ( accuracy )
    print

    if accuracy > best_acurracy[0]:
        best_acurracy[0:2] = accuracy, epoch
    return best_acurracy

def main():
    torch.manual_seed ( args.seed )
    torch.cuda.manual_seed_all ( args.seed )

    dataset = dataloader.dataset ( args, phase='train' )
    train_loader = DataLoader ( dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                pin_memory=True )
    dataset = dataloader.dataset ( args, phase='valid' )
    valid_loader = DataLoader ( dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True )
    print args.model
    if args.model == 'densenet169':
        net = densenet169(pretrained = args.pretrain)
    elif args.model == 'densenet161':
        net = densenet161 (  )
    elif args.model == 'densenet121':
        net = densenet121 (  )
    elif args.model == 'densenet201':
        net = densenet201 (  )
    elif args.model == 'resnext101':
        net = resnext101 ( num_classes = args.num_classes )
    elif args.model == 'resnext152':
        net = resnext152 ( num_classes = args.num_classes )
    elif args.model == 'se_resnet152':
        net = se_resnet152 ( num_classes = args.num_classes, pretrained = None )
    elif args.model == 'se_resnext101_32x4d':
        net = se_resnext101_32x4d ( num_classes = args.num_classes )
    elif args.model == 'inceptionresnetv2':
        net = inceptionresnetv2 ( num_classes = args.num_classes,pretrained = None )
    elif args.model == 'inceptionv4':
        net = inceptionv4 ( num_classes = args.num_classes, pretrained = None )
    elif args.model == 'dpn131':
        net = dpn131 ( num_classes = args.num_classes, pretrained = None )

    net = torch.nn.DataParallel ( net, device_ids=[0, 1, 2] )
    net = net.cuda ( )
    loss = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam ( net.parameters ( ), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-5 )
    best_acurracy = [0, 0]

    for epoch in range ( args.epochs ):
        train ( train_loader, net, loss, epoch, optimizer )
        torch.cuda.empty_cache ( )
        best_acurracy = test ( valid_loader, net, loss, epoch, best_acurracy )
        torch.cuda.empty_cache ( )

    print('best iou', best_acurracy)
    with open ( args.data_dir + args.result_file, 'a' ) as f:
        f.write ( args.model )
        f.write ( '\n' )
        f.write ( str(args.batch_size) )
        f.write ( '\n' )
        f.write ( str ( args.pretrain ) )
        f.write ( '\n' )
        f.write ( 'best accuracy' + ' ' + str ( best_acurracy[0] ) + str ( best_acurracy[1] ) )
        f.write ( '\n' )


if __name__ == '__main__':
    main()