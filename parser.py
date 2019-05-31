# coding=utf8

import argparse

parser = argparse.ArgumentParser(description = '')

parser.add_argument(
        '--data_dir',
        type=str,
        default='/DATASET/2019 百度大数据/初赛赛题/train/',
        help=''
        )

parser.add_argument(
        '--visit_dir',
        type=str,
        default='/DATASET/2019 百度大数据/初赛赛题/train_visit/',
        help=''
        )

parser.add_argument(
        '--train',
        type=str,
        default='train_1.json',
        help=''
        )

parser.add_argument(
        '--valid',
        type=str,
        default='valid_1.json',
        help=''
        )

parser.add_argument(
        '--result-file',
        type=str,
        default='result.txt',
        help='result files directory'
        )

parser.add_argument(
        '--model',
        '-m',
        metavar='MODEL',
        type=str,
        default='densenet169',
        help='densenet169' or 'densenet121' or 'densenet201' or 'densenet161' or 'resnext101' or 'resnext152'
        or 'se_resnet152' or 'se_resnext101_32x4d' or 'inceptionresnetv2' or 'inceptionv4' or 'dpn131'
        )

parser.add_argument('--input_channel',
        default=3,
        type=int,
        metavar='N',
        help='input size '
        )

parser.add_argument('--num_classes',
        default=9,
        type=int,
        metavar='N',
        help=''
        )

parser.add_argument('--batch_size',
        default= 192,
        type=int,
        metavar='N',
        )

parser.add_argument('-j',
        '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')


parser.add_argument('--epochs',
        default=200,
        type=int,
        metavar='N',
        )

parser.add_argument('--lr',
        '--learning-rate',
        default=0.001,
        type=float,
        metavar='LR',
        help='initial learning rate'
        )


parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="manual seed"
        )

parser.add_argument('--data-augmentation',
        default=True,
        type=str,
        metavar='N',
        help='if data augmentation'
        )
parser.add_argument('--pretrain',
        default=True,
        type=str,
        metavar='N',
        help='if data augmentation'
        )

# visit

parser.add_argument(
        '--day',
        type=int,
        default=182,
        help="manual seed"
        )

parser.add_argument(
        '--hour',
        type=int,
        default=24,
        help="manual seed"
        )

args = parser.parse_args()