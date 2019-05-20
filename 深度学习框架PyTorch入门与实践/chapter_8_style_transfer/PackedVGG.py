# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        features=list(vgg16(pretrained=True).features)[:23]
        self.features=nn.ModuleList(features).eval()

    def forward(self,x):
        results=[]
        #features的第3,8,15,22层分别是relu1_2,relu2_2,relu3_3,relu4_3
        for i ,model in enumerate(self.features):
            x=model(x)
            if i in{3,8,15,22}:
                results.append(x)

        vgg_outputs=namedtuple('VggOutputs',['relu1_2','relu2_2',
                                             'relu3_3','relu4_3'])
        return vgg_outputs
