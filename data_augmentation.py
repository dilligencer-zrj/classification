# coding=utf-8
import random
import numpy as np
import cv2
from scipy.ndimage import zoom
from skimage import transform
from matplotlib import pyplot as plt
plt.switch_backend ( 'TkAgg' )


def FlipH(img):
    return np.fliplr ( img )


def FlipV(img):
    return np.flipud ( img )


def Rotate(img, k):
    return np.rot90(img,k = k, axes=(0,1))

def Blur(img):
    '''
    加噪声
    :param img:
    :return:
    '''
    value =  np.random.randint ( 3, 6 )
    img = cv2.blur ( img, (value, value) )
    return img

def scale(img):

    '''
    从原图中以一定比例（0.7-1）crop出一块区域，再将这块区域放大到原图的大小

    '''
    scale_factor = random.uniform(0.7,1)
    crop_size_h = int(img.shape[0]*scale_factor)
    crop_size_w = int ( img.shape[1] * scale_factor )
    min_range_h = img.shape[0] - crop_size_h
    min_range_w = img.shape[1] - crop_size_w
    min_h = np.random.randint(0,min_range_h)
    min_w = np.random.randint ( 0, min_range_w )
    crop_img = img[min_h:(min_h+crop_size_h),min_w:(min_w+crop_size_w):,]
    upsample_factor = img.shape[0] * 1.0  / crop_img.shape[0]
    img = zoom(crop_img,[upsample_factor,upsample_factor,1])
    return img

def hsv_transform(img, hue_delta, sat_mult, val_mult):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = img_hsv.astype(np.float64)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

'''
随机hsv变换
hue_vari是色调变化比例的范围
sat_vari是饱和度变化比例的范围
val_vari是明度变化比例的范围
'''
def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)


def augmentation(img):
    original_img = img
    FlipH_flag = np.random.randint ( 0, 2 )
    if FlipH_flag:
        img = FlipH ( img )
    FlipV_flag = np.random.randint ( 0, 2 )
    if FlipV_flag:
        img = FlipV ( img )
    Rotate_flag = np.random.randint ( 0, 2 )
    if Rotate_flag:
        Rotate_time =  np.random.randint ( 1, 4 )
        img = Rotate(img,Rotate_time)
    Blur_flag = np.random.randint ( 0, 2 )
    if Blur_flag:
        img = Blur ( img )
    img = scale(img)
    img = random_hsv_transform(img,hue_vari= 10,sat_vari= 0.1, val_vari=0.1 )

    '''
    # 可视化数据增强前后的效果
    original_img_ = original_img[:, :, [2, 1, 0]]
    img_ = img[:, :, [2, 1, 0]]
    plt.subplot ( 121 )
    plt.imshow ( original_img_ )
    plt.subplot ( 122 )
    plt.imshow ( img_ )
    plt.show (  )
    '''

    return img

