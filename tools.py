#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import random
import argparse
from glob import glob
from sklearn.model_selection import train_test_split

def arg_parser():
    parser = argparse.ArgumentParser(
        prog='data dir',
        formatter_class=argparse.RawTextHelpFormatter,
        description='settings',
    )

    parser.add_argument (
        '--pic_dir',
        dest='pic_dir',
        default= '/DATASET/2019 百度大数据/初赛赛题/train/',
        type=str,
        help='original',
    )

    parser.add_argument (
        '--visit_dir',
        dest='visit_dir',
        default='/DATASET/2019 百度大数据/初赛赛题/train_visit/',
        type=str,
        help='original',
    )

    parser.add_argument (
        '-- ratio',
        dest='ratio',
        default=0.2,
        type=int,
        help='original',
    )

    parser.add_argument (
        '-- k_fold',
        dest='k_fold',
        default= 5,
        type=int,
        help='original',
    )

    return parser

def mywritejson(save_path,content):
    content=json.dumps(content,indent=4,ensure_ascii=False)
    with open(save_path,'w') as f:
        f.write(content)


def train_val_split(args):
    for k in range(1,args.k_fold+1):
        train_list = []
        val_list = []
        randnum = random.randint(1,100)
        category = ['001','002','003','004','005','006','007','008','009']
        for c in category:
            file_list = glob(args.pic_dir + c + '/' + '*.jpg')
            train,val = train_test_split(file_list,test_size=args.ratio,random_state=randnum)
            train_list += train
            val_list += val
        mywritejson(args.pic_dir + 'train_' + str(k) + '.json', train_list)
        mywritejson(args.pic_dir + 'valid_' + str(k) + '.json',val_list)


def visit_info_statistic(args):
    date = []
    users =[]
    visit_file_list = glob(args.visit_dir + '*.txt')
    for visit_file in visit_file_list:
        with open(visit_file,'r') as f:
            visit_info_list = f.readlines()
            for visit_info in visit_info_list:
                # user = visit_info.split('\t')[0]
                # if user not in users:
                #     users.append(user)
                # users.append(user)
                time = visit_info.split('\t')[1].split(',')
                for i in time:
                    if i.split('&')[0] not in date:
                        date.append(i.split('&')[0])
    date = sorted(date)
    print date
    print len(date)





if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    # train_val_split(args)
    visit_info_statistic(args)