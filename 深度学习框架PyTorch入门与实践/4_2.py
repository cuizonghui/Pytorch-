# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import torch as t
from torch import nn
from torch.autograd import Variable


# to_tensor=ToTensor()
# to_pil=ToPILImage()
# lena=Image.open('./lena.jpeg')
# lena.show()
# gray_lena=lena.convert('L')
# gray_lena.show()
# # print(lena)
# # fig=plt.figure()
# # plt.imshow(lena)
# # plt.show()
#
#
# input=to_tensor(gray_lena).unsqueeze(0)
#
# kernel=t.ones(3,3)/-9.
# print(kernel)
# kernel[1][1]=1
# conv=nn.Conv2d(1,1,(3,3),1,bias=False)
# conv.weight.data=kernel.view(1,1,3,3)
#
# out=conv(Variable(input))
# to_pil(out.data.squeeze(0)).show()
#
# pool_out=nn.AvgPool2d(2,2)(Variable(input))
# to_pil(pool_out.data.squeeze(0)).show()


# input=Variable(t.randn(2,3))
# linear=nn.Linear(3,4)
# h=linear(input)
# print(h)
#
# bn=nn.BatchNorm1d(4)
# bn.weight.data=t.ones(4)*4
# bn.bias.data=t.zeros(4)
# bn_out=bn(h)
# print(bn_out)
# print(bn_out.mean(0),bn_out.var(0,unbiased=False))
# print(bn_out)
# dropout=nn.Dropout(0.5)
# o=dropout(bn_out)
# print(o)
#
# relu=nn.ReLU(inplace=True)
#
# input=Variable(t.randn(2,3))
# print(input)
# output=relu(input)
# print(output)


# Sequential的三种写法
# from collections import OrderedDict
#
# net1=nn.Sequential()
# net1.add_module('conv',nn.Conv2d(3,3,3))
# net1.add_module('batchnorm',nn.BatchNorm2d(3))
# net1.add_module('activation_layer',nn.ReLU())
#
# net2=nn.Sequential(
#     nn.Conv2d(3,3,3),
#     nn.BatchNorm2d(3),
#     nn.ReLU()
# )
#
# net3=nn.Sequential(OrderedDict([
#     ('conv1',nn.Conv2d(3,3,3)),
#     ('bn1',nn.BatchNorm2d(3)),
#     ('relu',nn.ReLU())
# ]))
#
# print('net1:',net1)
# print('net2:',net2)
# print('net3:',net3)
#
# modellist=nn.ModuleList([nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2)])
# input=Variable(t.randn(1,3))
# for model in modellist:
#     input=model(input)
#


t.manual_seed(400)
input = Variable(t.randn(2, 3, 4))
lstm = nn.LSTM(4, 3, 1)
h0 = Variable(t.randn(1, 3, 3))
c0 = Variable(t.randn(1, 3, 3))
out, hn = lstm(input, (h0, c0))
print(out)
