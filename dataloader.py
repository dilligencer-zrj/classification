import cv2
import json
import numpy as np
import datetime
from torch.utils.data import DataLoader, Dataset

from data_augmentation import augmentation

def visit_transform(img_root, args):
    name = img_root.split('/')[-1].replace('.jpg','.txt')
    visit_path = args.visit_dir + name
    visit_array = np.zeros((args.day,args.hour))
    with open(visit_path,'r') as f:
        visit_info_list = f.readlines ( )
        for visit_info in visit_info_list:
            time = visit_info.split ( '\t' )[1].split ( ',' )
            for i in time:
                date = i.split('&')[0]
                year = int(date[:4])
                month = int ( date[4:6] )
                day = int ( date[6:] )
                current_date =  datetime.datetime(year,month,day)
                base_date= datetime.datetime(2018,10,01)
                difference = int ( (current_date - base_date).days )
                hours = i.split('&')[1].split('|')
                for hour in hours:
                    hour = int(hour)
                    visit_array[difference][hour] += 1
    max_value = np.max(visit_array)
    min_value = np.min(visit_array)
    visit_array = (visit_array - min_value) * 1.0 / (max_value - min_value)

    return visit_array


class dataset(Dataset):
    def __init__(self,args, phase = 'train'):
        super(dataset,self).__init__()
        self.args = args
        self.phase = phase
        if phase == 'train':
            self.data_list = json.load(open(args.data_dir + args.train))
        elif phase == 'valid':
            self.data_list = json.load ( open ( args.data_dir + args.valid ) )

    def __getitem__(self, idx):
        img_root = self.data_list[idx]
        visit_array = visit_transform(img_root,self.args)
        name = img_root.split('/')[-1]
        label = int(img_root.split('/')[-2]) -1
        img = cv2.imread(img_root)
        if self.phase == 'train' and self.args.data_augmentation:
            img = augmentation(img)
        img = img / 255.0
        img= np.transpose(img,(2,0,1))
        return np.array(img), np.array(visit_array), label, name


    def __len__(self):
        return len ( self.data_list )