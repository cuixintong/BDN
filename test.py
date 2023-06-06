import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from datasets.pretrain_datasets import TrainData, ValData, TestData
from models.BDN import BDN
from models.FFA import FFANet
from utils import to_psnr, print_log, validation, adjust_learning_rate
import numpy as np
import os
from PIL import Image

test_batch_size = 1

epoch = 14

test_data_dir = 'D:/app/pycharm/space/dehaze/FFA-Net'
    
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


crop_size = 256#输入图片的大小

gps=3
blocks=19
net = BDN(gps=gps,blocks=blocks)
net = nn.DataParallel(net, device_ids=device_ids)


net.module.load_state_dict(torch.load('output/haze_current19.pth'))


net.eval()


test_data_loader = DataLoader(TestData(test_data_dir+'/RESIDE/SOTS/SOTS/indoor/nyuhaze500/',train=False,size=crop_size), batch_size=test_batch_size, shuffle=False, num_workers=0)
test_data_loader_dcp = DataLoader(TestData(test_data_dir+'/RESIDE/SOTS/SOTS/indoor/nyuhaze500_dcp/',train=False,size=crop_size), batch_size=test_batch_size, shuffle=False, num_workers=0)

# ITS_test_loader=DataLoader(dataset=TrainData(path+'/RESIDE/ITS/ITS/ITS/val/',train=False,size='whole_img'),batch_size=1,shuffle=False)



output_dir = 'output/base_JEPG8/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
with torch.no_grad():
    # for batch_id, val_data in enumerate(test_data_loader):
    for batch_id, test_data in enumerate(zip(test_data_loader,test_data_loader_dcp)):
        if batch_id > 150:
            break
        # haze, name = val_data # For GCA

        haze, gt, img = test_data[0]
        haze_dcp, gt_dcp, img_dcp = test_data[1]

        haze = haze.to(device)
        haze_dcp = haze_dcp.to(device)

        print(batch_id, 'BEGIN!')

        # pred = net(haze, 0, True, False) # For GCA

        # return out, out_J, out_T, out_A, out_I
        # _, pred, T, A, I = net(haze, haze_dcp, True) # For FFA and MSBDN
        out_J, out_T, out_A, out_I = net(haze, haze_dcp, True) # For FFA and MSBDN

        ### GCA ###
        # dehaze = pred.float().round().clamp(0, 255)
        # out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
        # out_img.save(output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###########
        
        ### FFA & MSBDN ###
        ts = torch.squeeze(out_J.clamp(0, 1).cpu())
        vutils.save_image(ts, output_dir +'_MyModel_{}.png'.format(batch_id))
        # vutils.save_image(ts, output_dir + img[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###################
