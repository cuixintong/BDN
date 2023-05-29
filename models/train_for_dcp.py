#添加必要的库
import os
from DCP import *

def main():

    path = "D:\\app\pycharm\space\dehaze\FFA-Net\RESIDE\SOTS\SOTS\indoor\\nyuhaze500/"   #图像读取地址
    savepath = "D:\\app\pycharm\space\dehaze\FFA-Net\RESIDE\SOTS\SOTS\indoor\\nyuhaze500_dcp/haze/"  # 图像保存地址
    filelist = os.listdir(path)  # 打开对应的文件夹
    total_num = len(filelist)  #得到文件夹中图像的个数
    print(111)

    haze_imgs_dir = os.listdir(
        os.path.join(path, 'haze'))  # ['0001_01_0.9027.png', '0001_02_0.8096.png', ...]
    # 将有雾图像的名字（绝对名字）得到并存储到列表中
    haze_imgs = [os.path.join(path, 'haze', img) for img in
                      haze_imgs_dir]
    print(haze_imgs)

    for img_name in haze_imgs_dir:
        DCP(img_name,os.path.join(path,'haze'),savepath)


if __name__ == "__main__":
    print(111)

    main()