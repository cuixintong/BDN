import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import feature


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads): # q的维度，k的维度，输出的维度 维度=通道数
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, mask=None):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = 2
        # split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        ## score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(3, 4))  # [h, N, T_q, T_k]
        # scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)

        ## mask
        if mask is not None:
            ## mask:  [N, T_k] --> [h, N, T_q, T_k]
            mask = mask.feature1(1).feature1(0).repeat(self.num_heads, 1, querys.shape[2], 1)
            scores = scores.masked_fill(mask, -np.inf)
        scores = F.softmax(scores, dim=3)

        ## out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        out = F.avg_pool2d(out, (out.shape[2], out.shape[3]))

        return out, scores

class BlockUNet1(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, relu=False, drop=False, bn=True):
        super(BlockUNet1, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

        self.dropout = nn.Dropout2d(0.5)
        self.batch = nn.InstanceNorm2d(out_channels)

        self.upsample = upsample
        self.relu = relu
        self.drop = drop
        self.bn = bn

    def forward(self, x):
        if self.relu == True:
            y = F.relu(x)
        elif self.relu == False:
            y = F.leaky_relu(x, 0.2)
        if self.upsample == True:
            y = self.deconv(y)
            if self.bn == True:
                y = self.batch(y)
            if self.drop == True:
                y = self.dropout(y)

        elif self.upsample == False:
            y = self.conv(y)
            if self.bn == True:
                y = self.batch(y)
            if self.drop == True:
                y = self.dropout(y)

        return y

class G2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(G2, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = 8, kernel_size = 2,
                              stride = 2,padding = 0, bias=False)
        self.layer1 = BlockUNet1(8, 16)
        self.layer2 = BlockUNet1(16, 32)
        self.layer3 = BlockUNet1(32, 64)
        self.layer4 = BlockUNet1(64, 64)
        self.layer5 = BlockUNet1(64, 64)
        self.layer6 = BlockUNet1(64, 64)
        self.layer7 = BlockUNet1(64, 64)
        self.dlayer7 = BlockUNet1(64, 64, True, True, True, False)
        self.dlayer6 = BlockUNet1(128, 64, True, True, True)
        self.dlayer5 = BlockUNet1(128, 64, True, True, True)
        self.dlayer4 = BlockUNet1(128, 64, True, True)
        self.dlayer3 = BlockUNet1(128, 32, True, True)
        self.dlayer2 = BlockUNet1(64, 16, True, True)
        self.dlayer1 = BlockUNet1(32, 8, True, True)
        self.relu = nn.ReLU()
        self.dconv = nn.ConvTranspose2d(16, out_channels, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # print(x)

        x = x.to(torch.float32)
        y1 = self.conv(x)
        # print(y1)
        # print("111--------------------------------")
        y2 = self.layer1(y1)
        y3 = self.layer2(y2)
        y4 = self.layer3(y3)
        y5 = self.layer4(y4)
        y6 = self.layer5(y5)
        y7 = self.layer6(y6)
        # y8 = self.layer7(y7)

        dy7 = self.dlayer7(y7)
        concat6 = torch.cat([dy7, y6], 1)
        dy6 = self.dlayer5(concat6)
        concat5 = torch.cat([dy6, y5], 1)
        dy5 = self.dlayer4(concat5)
        concat4 = torch.cat([dy5, y4], 1)
        dy4 = self.dlayer3(concat4)
        concat3 = torch.cat([dy4, y3], 1)
        dy3 = self.dlayer2(concat3)
        concat2 = torch.cat([dy3, y2], 1)
        dy2 = self.dlayer1(concat2)
        concat1 = torch.cat([dy2, y1], 1)


        # dy8 = self.dlayer7(y8)
        # concat7 = torch.cat([dy8, y7], 1)
        # dy7 = self.dlayer6(concat7)
        # concat6 = torch.cat([dy7, y6], 1)
        # dy6 = self.dlayer5(concat6)
        # concat5 = torch.cat([dy6, y5], 1)
        # dy5 = self.dlayer4(concat5)
        # concat4 = torch.cat([dy5, y4], 1)
        # dy4 = self.dlayer3(concat4)
        # concat3 = torch.cat([dy4, y3], 1)
        # dy3 = self.dlayer2(concat3)
        # concat2 = torch.cat([dy3, y2], 1)
        # dy2 = self.dlayer1(concat2)
        # concat1 = torch.cat([dy2, y1], 1)

        out = self.relu(concat1)
        out = self.dconv(out)
        out = self.lrelu(out)

        return out
        # return F.avg_pool2d(out, (out.shape[2], out.shape[3]))
class MVSA(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MVSA, self).__init__()
        self.FC = G2(3, 3)
        self.attention = MultiHeadAttention(256,256, 256, 1)

    def forward(self, feature1, mask=None):
        Fd = self.FC(feature1)
        A, scores = self.attention(Fd, Fd)

        return A, scores


if __name__ == '__main__':
    mvsa = MVSA(3, 3)
    feature1 = cv2.imread("../image/0001_0.8_0.2_dehaze.png")
    feature1 = torch.from_numpy(feature1)
    print(feature1.shape)
    feature1 = feature1.permute(2, 0, 1)
    print(feature1.shape)
    print(type(feature1))
    feature1 = torch.unsqueeze(feature1, 0)
    print(feature1.shape)

    from datasets.pretrain_datasets import *

    haze,gt = next(iter(ITS_train_loader))
    # print(haze[0])
    # haze = haze[0]
    # haze = torch.unsqueeze(haze, 0)

    # feature1 = np.zeros((4,4,3))
    # feature1 = torch.from_numpy(feature1)
    # feature1 = feature1.permute(2, 0, 1)
    # print(feature1.shape)
    #
    # feature2 = feature1

    out,score = mvsa(haze, gt)
    cv2.imshow('put',out.detach().numpy());

    # print(mvsa)

