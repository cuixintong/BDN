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

epoch = 14

test_data_dir = 'D:/app/pycharm/space/dehaze/FFA-Net'
    
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


crop_size = 256#输入图片的大小

gps=3
blocks=19
net = BDN(gps=gps,blocks=blocks)
net = nn.DataParallel(net, device_ids=device_ids)


net.load_state_dict(torch.load('output/haze_current1.pth'))


net.eval()


val_data_loader_dcp = DataLoader(ValData(test_data_dir+'/RESIDE/ITS/ITS/ITS/val_dcp/',train=False,size=crop_size), batch_size=val_batch_size, shuffle=False, num_workers=0)



output_dir = '/output/base_JEPG8/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
with torch.no_grad():
    for batch_id, val_data in enumerate(test_data_loader):
        if batch_id > 150:
            break
        # haze, name = val_data # For GCA
        haze, haze_A, name = val_data # For FFA and MSBDN
        haze.to(device)
        
        print(batch_id, 'BEGIN!')

        # pred = net(haze, 0, True, False) # For GCA
        _, pred, T, A, I = net(haze, haze_A, True) # For FFA and MSBDN
        
        ### GCA ###
        # dehaze = pred.float().round().clamp(0, 255)
        # out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
        # out_img.save(output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###########
        
        ### FFA & MSBDN ###
        ts = torch.squeeze(pred.clamp(0, 1).cpu())
        vutils.save_image(ts, output_dir + name[0].split('.')[0] + '_MyModel_{}.png'.format(batch_id))
        ###################
