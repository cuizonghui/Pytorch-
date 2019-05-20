# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):

    def __init__(self,in_channel,out_channel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel))

        self.right=shortcut

    def forward(self, input):
        out=self.left(input)
        residual=input if self.right is None else self.right(input)
        out+=residual
        return F.relu(out)

class ResNet34(BasicModule):

    def __init__(self,num_classes=2):
        super(ResNet34,self).__init__()
        self.model_name='resnet34'

        #前几层图像转换
        self.pre=nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )

        self.layer1=self._make_layer(64,128,3)
        self.layer2=self._make_layer(128,256,4,stride=2)
        self.layer3=self._make_layer(256,512,6,stride=2)
        self.layer4=self._make_layer(512,512,3,stride=2)

        self.fc=nn.Linear(512,num_classes)

    def _make_layer(self,in_channel,out_channel,block_num,stride=1):
        shortcut=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,1,stride,bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers=[]

        layers.append(ResidualBlock(in_channel,out_channel,stride,shortcut))
        for i in range(block_num):
            layers.append(ResidualBlock(out_channel,out_channel))
        return nn.Sequential(*layers)

    def forward(self,input):
        input=self.pre(input)
        input=self.layer1(input)
        input=self.layer2(input)
        input=self.layer3(input)
        input=self.layer4(input)
        input=F.avg_pool2d(input,7)
        input=input.view(input.size(0),-1)
        return self.fc(input)