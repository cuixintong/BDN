import cv2
import torch.utils.data as data
import os
from PIL import Image
from random import randrange
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
# from utils import edge_compute
from models.options import opt

import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
import random


class TrainData(data.Dataset):

    def __init__(self, train_data_dir,train,size,format='.png'):

        super(TrainData, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        # self.haze_imgs_dir=os.listdir(os.path.join(path))
        # 得到有雾图的文件夹

        self.haze_imgs_dir = os.listdir(os.path.join(train_data_dir, 'haze'))  # ['0001_01_0.9027.png', '0001_02_0.8096.png', ...]
        # 将有雾图像的名字（绝对名字）得到并存储到列表中
        self.haze_imgs = [os.path.join(train_data_dir, 'haze', img) for img in
                          self.haze_imgs_dir]  # ['D:/app/pycharm/space/dehaze/FFA-Net/RESIDE/ITS/ITS/ITS/train/0001_01_0.9027.png', ...]
        # self.haze_imgs=[os.path.join(path,'haze',img) for img in self.haze_imgs_dir]
        # 得到清晰图像的文件夹
        self.clear_dir = os.path.join(train_data_dir, 'clear')


    # def __getitem__(self, index):
    #
    #     crop_width = self.crop_size
    #     crop_height = self.crop_size
    #
    #     haze_name = self.haze_names[index]
    #     gt_name = haze_name.split('_')[0] + '.png'
    #     haze_img = Image.open(self.haze_dir + haze_name).convert('RGB')
    #     gt_img = Image.open(self.gt_dir + gt_name).convert('RGB')
    #
    #     width, height = haze_img.size
    #
    #     if width < crop_width or height < crop_height:
    #         if width < height:
    #             haze_img = haze_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
    #             gt_img = gt_img.resize((260, (int)(height * 260/ width)), Image.ANTIALIAS)
    #         elif width >= height:
    #             haze_img = haze_img.resize(((int)(width * 260/ height), 260), Image.ANTIALIAS)
    #             gt_img = gt_img.resize(((int)(width * 260 / height), 260), Image.ANTIALIAS)
    #
    #         width, height = haze_img.size
    #     # --- random crop --- #
    #     x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
    #     haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
    #     gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
    #
    #     transform_haze = Compose([
    #         ToTensor(),
    #         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])
    #     transform_gt = Compose([
    #         ToTensor()
    #     ])
    #
    #     haze = transform_haze(haze_crop_img)
    #     gt = transform_gt(gt_crop_img)
    #
    #     if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
    #         raise Exception('Bad image channel: {}'.format(gt_name))
    #
    #     return haze, gt

    def __getitem__(self, index):
        #得到一张图像
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id = img.split('\\')[-1].split('_')[0]#windows
        # id=img.split('/')[-1].split('_')[0]#linux
        clear_name=id+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        #对clear进行中心裁剪
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        # print(type(self.size))
        # if isinstance(self.size,str):
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        # print(haze.shape,clear.shape)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target=tfs.ToTensor()(target)
        return  data ,target

    def __len__(self):
        return len(self.haze_imgs)

class ValData(data.Dataset):
    def  __init__(self, val_data_dir,train,size,format='.png'):
        super(ValData, self).__init__()

        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format

        self.haze_imgs_dir = os.listdir(
            os.path.join(val_data_dir, 'haze'))  # ['0001_01_0.9027.png', '0001_02_0.8096.png', ...]
        # 将有雾图像的名字（绝对名字）得到并存储到列表中
        self.haze_imgs = [os.path.join(val_data_dir, 'haze', img) for img in
                          self.haze_imgs_dir]  # ['D:/app/pycharm/space/dehaze/FFA-Net/RESIDE/ITS/ITS/ITS/train/0001_01_0.9027.png', ...]
        # self.haze_imgs=[os.path.join(path,'haze',img) for img in self.haze_imgs_dir]
        # 得到清晰图像的文件夹
        self.clear_dir = os.path.join(val_data_dir, 'clear')


    def get_images(self, index):
        # 得到一张图像
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split('\\')[-1].split('_')[0]  # windows
        # id=img.split('/')[-1].split('_')[0]#linux
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        # 对clear进行中心裁剪
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        #print(type(self.size))
        # if isinstance(self.size,str):
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        # print(haze.shape,clear.shape)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear, img

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_imgs)


class TestData(data.Dataset):
    # def __init__(self, val_data_dir):
    #     super().__init__()
    #
    #     self.haze_dir = val_data_dir
    #     #self.gt_dir = val_data_dir + 'clear/'
    #     with open('/configs/record.txt', "r") as file:
    #         self.haze_names = file.readlines()

    def __init__(self, test_data_dir, train, size, format='.png'):

        super(TrainData, self).__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        # self.haze_imgs_dir=os.listdir(os.path.join(path))
        # 得到有雾图的文件夹

        self.haze_imgs_dir = os.listdir(
            os.path.join(test_data_dir, 'haze'))  # ['0001_01_0.9027.png', '0001_02_0.8096.png', ...]
        # 将有雾图像的名字（绝对名字）得到并存储到列表中
        self.haze_imgs = [os.path.join(test_data_dir, 'haze', img) for img in
                          self.haze_imgs_dir]  # ['D:/app/pycharm/space/dehaze/FFA-Net/RESIDE/ITS/ITS/ITS/train/0001_01_0.9027.png', ...]
        # self.haze_imgs=[os.path.join(path,'haze',img) for img in self.haze_imgs_dir]
        # 得到清晰图像的文件夹

    def __getitem__(self, index):
        # 得到一张图像
        haze = Image.open(self.haze_imgs[index])
        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, 20000)
                haze = Image.open(self.haze_imgs[index])
        img = self.haze_imgs[index]
        id = img.split('\\')[-1].split('_')[0]  # windows
        # id=img.split('/')[-1].split('_')[0]#linux
        clear_name = id + self.format
        clear = Image.open(os.path.join(self.clear_dir, clear_name))
        # 对clear进行中心裁剪
        clear = tfs.CenterCrop(haze.size[::-1])(clear)
        print(type(self.size))
        # if isinstance(self.size,str):
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        # print(haze.shape,clear.shape)
        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)
    
def edge_compute(x):
    
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    
    return y

crop_size = opt.crop_size
BS = opt.bs

path = "D:\\app\pycharm\space\dehaze\FFA-Net"


# def __init__(self, train_data_dir, train, size, format='.png'):


ITS_train_loader=DataLoader(dataset=TrainData(path+'/RESIDE/ITS/ITS/ITS/train/',train=True,size=crop_size),batch_size=BS,shuffle=True)
ITS_test_loader=DataLoader(dataset=TrainData(path+'/RESIDE/ITS/ITS/ITS/val/',train=False,size='whole_img'),batch_size=1,shuffle=False)

OTS_train_loader=DataLoader(dataset=TrainData(path+'/RESIDE/SOTS/SOTS/indoor/nyuhaze500',train=True,size=crop_size,format='.png'),batch_size=BS,shuffle=True)
OTS_test_loader=DataLoader(dataset=TrainData(path+'/RESIDE/SOTS/SOTS/outdoor/outdoor',train=False,size='whole_img',format='.png'),batch_size=1,shuffle=False)

if __name__ == '__main__':

    # haze,gt = next(iter(ITS_train_loader))
    # print(haze[0].shape)
    # haze = haze[0].permute(1,2,0)
    # haze = haze.detach().numpy()
    # cv2.imshow("haze",haze)
    # cv2.imwrite("haze.png",haze)

    dataset=TrainData(path+'/RESIDE/ITS/ITS/ITS/train/',train=True,size=crop_size)
    haze,clear = dataset.__getitem__(1)
    haze = haze.permute(1,2,0).numpy()
    cv2.imshow("haze",haze)
    cv2.imwrite("haze.png",haze)

    # print(haze)
