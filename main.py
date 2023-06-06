import os
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import TrainData, ValData
from models import FFA
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
import math
from pdb import set_trace as bp
from models.BDN import BDN

#from perceptual import LossNetwork

# 学习率更新
def lr_schedule_cosdecay(t,T,init_lr=1e-4):
    lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
    return lr


# 初始化参数
lr=1e-4




crop_size = 256#输入图片的大小
# crop_size = [256, 256]#输入图片的大小
train_batch_size = 1
val_batch_size = 1
num_epochs = 20
category = 'indoor'

gps=3
blocks=19
net = BDN(gps=gps,blocks=blocks)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# 设置多卡训练

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# device_ids = [Id for Id in range(torch.cuda.device_count())]
# device_ids = [1, 2, 3]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.cuda.set_device('cuda:{}'.format(device_ids[0]))
net.to(device)
# 将模型用于多GPU训练，并行处理
# net = nn.DataParallel(net.cuda(), device_ids=device_ids)


path = "D:/app/pycharm/space/dehaze/FFA-Net"

#加载数据集
train_data_loader = DataLoader(TrainData(path+'/RESIDE/ITS/ITS/ITS/train/',train=True,size=crop_size), batch_size=train_batch_size, shuffle=True, num_workers=0)
val_data_loader = DataLoader(ValData(path+'/RESIDE/ITS/ITS/ITS/val/',train=False,size=crop_size), batch_size=val_batch_size, shuffle=False, num_workers=0)


print("DATALOADER DONE!")

#old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category)
#print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
# 使用cudnn加速
torch.backends.cudnn.benchmark = True
old_val_psnr = 0
all_T = 100000

# 开始训练
for epoch in range(num_epochs):
    psnr_list = []
    start_time = time.time()
    #adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):
        # print(batch_id)
        # if batch_id > 5000:
        if batch_id > 5:
            break
        # step_num = batch_id + epoch * 5000 + 1
        step_num = batch_id + epoch * 5 + 1
        lr=lr_schedule_cosdecay(step_num,all_T)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        haze, gt = train_data


        # haze = nn.DataParallel(haze, device_ids=device_ids)
        # gt = nn.DataParallel(gt, device_ids=device_ids)
        # haze_dcp = nn.DataParallel(haze_dcp, device_ids=device_ids)
        # gt_dcp = nn.DataParallel(gt_dcp, device_ids=device_ids)


        # haze = haze.cuda()
        # gt = gt.cuda()
        # haze_dcp = haze_dcp.cuda()
        # gt_dcp = gt_dcp.cuda()

        haze = haze.to(device)
        gt = gt.to(device)


        #dc = get_dark_channel(haze, 15)
        #A = get_atmosphere(haze, dc, 0.0001)
        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        J, T, A, I = net(haze)
        #s, v = get_SV_from_HSV(J)
        #CAP_loss = F.smooth_l1_loss(s, v)
        Rec_Loss1 = F.smooth_l1_loss(J, gt)
        Rec_Loss2 = F.smooth_l1_loss(I, haze)

        #perceptual_loss = loss_network(dehaze, gt)
        loss = Rec_Loss1 + Rec_Loss2

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(J, gt))

        #if not (batch_id % 100):
        print('Epoch: {}, Iteration: {}, Loss: {}, Rec_Loss1: {}, Rec_loss2: {}'.format(epoch, batch_id, loss, Rec_Loss1, Rec_Loss2))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), 'output/haze_current{}.pth'.format(epoch))
    # torch.save(net.module.state_dict(), '/output/haze_current{}.pth'.format(epoch))

    # --- Use the evaluation model in testing --- #
    net.eval()

    net_name = "BDNNET"
    val_psnr, val_ssim = validation(net, net_name, val_data_loader, device, category)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)

    # --- update the network weight --- #
    #if val_psnr >= old_val_psnr:
    #    torch.save(net.state_dict(), '{}_haze_best_{}_{}'.format(category, network_height, network_width))
    #    old_val_psnr = val_psnr