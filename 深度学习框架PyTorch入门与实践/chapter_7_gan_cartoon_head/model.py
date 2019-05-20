# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
from torch import nn


class NetG(nn.Module):
    '''
    生成器定义
    '''
    def __init__(self,opt):
        super(NetG,self).__init__()
        ngf=opt.ngf #生成器feature map数

        self.main=nn.Sequential(
            #输入nz维度的噪声,可认为式一个nz*1*1的feature map
            nn.ConvTranspose2d(opt.nz,ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),
            #上一步的输形状(ngf*8)x4x4

            nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            ##上一步的输入形状为(ngf*4)x8x8

            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            #上一步的输出形状为（ngf*2）x16*16

            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #上一步的输出形状为(ngf)x32x32

            nn.ConvTranspose2d(ngf,3,5,3,1,bias=False),
            nn.Tanh()
            #输出形状为xx96x96
        )

    def forward(self, input):
        return self.main(input)


class NetD(nn.Module):
    '''
    判别器定义
    '''

    def __init__(self,opt):
        super(NetD,self).__init__()
        ndf=opt.ndf
        self.main=nn.Sequential(
            #输入3x96x96
            nn.Conv2d(3,ndf,5,3,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            #输出(ndf)x32x32

            nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),
            #输出(npd*2)x16x16

            nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace=True),
            #输出(npd*4)x8x8


            nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True ),
            #输出(ndf*8)x4x4

            nn.Conv2d(ndf*8,1,4,1,0,bias=False),
            nn.Sigmoid()#输出一个数字概率
        )

    def forward(self, input):
        return self.main(input).view(-1)
