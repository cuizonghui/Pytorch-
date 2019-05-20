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
import torch as t
import numpy as np


class TransformerNet(nn.Module):

    def __init__(self):
        super(TransformerNet,self).__init__()

        #下卷积
        self.initial_layers=nn.Sequential(
            ConvLayer(3,32,kernel_size=9,stride=1),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(True),
            ConvLayer(32,64,kernel_size=3,stride=2),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(True),
            ConvLayer(64,128,kernel_size=3,stride=2),
            nn.InstanceNorm2d(128,affine=True),
            nn.ReLU(True)
      )


        #Residual layers(残差曾)
        self.res_layers=nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )


        #上卷积层
        self.upsample_layers=nn.Sequential(
            UpsampleConvLayer(128,64,kernel_size=3,stride=1,upsample=2),
            nn.InstanceNorm2d(64,affine=True),
            nn.ReLU(True),
            UpsampleConvLayer(64,32,kernel_size=3,stride=1,upsample=2),
            nn.InstanceNorm2d(32,affine=True),
            nn.ReLU(True),
            ConvLayer(32,3,kernel_size=9,stride=1)
        )
    def forward(self, input):
        input=self.initial_layers(input)
        input=self.res_layers(input)
        input=self.upsample_layers(input)
        return input


class ConvLayer(nn.Module):
    '''
    在卷积种增加reflectionpad 方式
    默认是补０，这里采用边界反射补充
    '''

    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(ConvLayer,self).__init__()
        reflection_padding=int(np.floor(kernel_size/2))
        self.reflection_pad=nn.ReflectionPad2d(reflection_padding)
        self.conv2d=nn.Conv2d(in_channels,out_channels,kernel_size,stride)

        def forward(self, input):
            out=self.reflection_pad(input)
            out=self.conv2d(out)
            return out

class UpsampleConvLayer(nn.Module):
    '''
    默认的卷积ｐａｄｄｉｎｇ是补０，这里使用边界反射补充
    先上采样，然后做一个卷积，这种效果更好
    '''
    def __init__(self,in_channels,out_channels,kernel_size,stride,upsample=None):
        super(UpsampleConvLayer,self).__init__()
        self.upsample=upsample
        reflection_padding=int(np.floor(kernel_size/2))
        self.reflection_pad=nn.ReflectionPad2d(reflection_padding)
        self.conv2d=nn.Conv2d(in_channels,out_channels,kernel_size,stride)

    def forward(self, input):
        if self.upsample:
            input=t.nn.functional.interpolate(input,scale_factor=self.upsample)


        out=self.reflection_pad(input)
        out=self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.conv1=ConvLayer(channels,channels,kernel_size=3,stride=1)
        self.in1=nn.InstanceNorm2d(channels,affine=True)
        self.conv2=ConvLayer(channels,channels,kernel_size=3,stride=1)
        self.int2=nn.InstanceNorm2d(channels,affine=True)
        self.relu=nn.ReLU()

    def forward(self, input):
        residual=input
        out=self.relu(self.in1(self.conv1(input)))
        out=self.in2(self.conv2(out))
        out+=residual
        return out