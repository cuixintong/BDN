import torch
import torch.nn as nn


from datasets.pretrain_datasets import ITS_train_loader


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=bias)
    
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1=conv(dim, dim, kernel_size, bias=True)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=conv(dim,dim,kernel_size,bias=True)
        self.calayer=CALayer(dim)
        self.palayer=PALayer(dim)
    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x 
        res=self.conv2(res)
        res=self.calayer(res)
        res=self.palayer(res)
        res += x 
        return res
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [ Block(conv, dim, kernel_size)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FFANet(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(FFANet, self).__init__()
        self.gps=gps
        self.dim=64
        kernel_size=3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca=nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer=PALayer(self.dim)

        self.conv_J_1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.conv_J_2 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        self.conv_T_1 = nn.Conv2d(64, 16, 3, 1, 1, bias=False)
        self.conv_T_2 = nn.Conv2d(16, 1, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(6, 3, 3, 1, 1, bias=False)
        
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1, x3, x2=0, Val=False):

        x1 = torch.cat([x1, x3], 1)
        x1 = self.conv2(x1) # 这里需要x1是[batch,3,256,256]
        x = self.pre(x1)
        res1=self.g1(x)
        res2=self.g2(res1)
        res3=self.g3(res2)
        w=self.ca(torch.cat([res1,res2,res3],dim=1))
        w=w.view(-1,self.gps,self.dim)[:,:,:,None,None]
        out=w[:,0,::]*res1+w[:,1,::]*res2+w[:,2,::]*res3
        out=self.palayer(out)

        # # 分别得到J A t(x) 计算得到 I
        # out_J = self.conv_J_1(out)
        # out_J = self.conv_J_2(out_J)
        # out_J = out_J + x1
        out_T = self.conv_T_1(out)
        out_T = self.conv_T_2(out_T)
        
        # if Val == False:
        #     out_A = self.ANet(x1)
        # else:
        #     out_A = self.ANet(x2)
            
        # out_I = out_T * out_J + (1 - out_T) * out_A
        #x=self.post(out)
        return out,  out_T
        # return out,  out_T, out_A

if __name__ == "__main__":
    # net = FFANet(gps = 3, blocks = 19)
    # print(net)
    #
    haze,gt = next(iter(ITS_train_loader))
    net = FFANet(gps = 3, blocks = 19)
    out, out_T = net(haze, gt)
