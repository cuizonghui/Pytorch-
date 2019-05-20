# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------\

# import torch as t
# from matplotlib import pyplot as plt
# import numpy as np




# 设置随机数种子，保证在不同计算机上运行下边的输出一致
# t.manual_seed(400)

#
# def get_fake_data(batch_size=8):
#     """
#     产生随机数据：ｙ＝x*2+3,加上一些噪声
#     :param batch_size:
#     :return:
#     """
#     x = t.rand(batch_size, 1) * 20
#     y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
#     return x, y
#
#
# # x, y = get_fake_data()
# # # x_np=x.squeeze().numpy()
# # # y_np=y.squeeze().numpy()
# #
# # # x_np_sort_indices=np.argsort(x_np)
# # #
# # # print(x_np[x_np_sort_indices])
# # # print(y_np[x_np_sort_indices])
# #
# # plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
# # plt.show()
#
# w=t.rand(1,1)
# b=t.zeros(1,1)
# lr=0.001
#
#
#
#
# for i in range(200000):
#     x,y=get_fake_data()
#
#     y_pred=x.mm(w)+b.expand_as(y)
#
#
#
#     loss=0.5*(y_pred-y)**2
#     loss=loss.sum()
#
#     dloss=1
#     dy_pred=dloss*(y_pred-y)
#
#     dw=x.t().mm(dy_pred)
#     db=dy_pred.sum()
#
#     w.sub_(lr*dw)
#     b.sub_(lr*db)
#
#     if i%1000==0:
#         x1 = t.arange(0, 20).view(-1, 1).float()
#         print(x.shape)
#         y=x1.mm(w)+b.expand_as(x1)
#         plt.plot(x1.numpy(),y.numpy())
#
#         x2,y2=get_fake_data(batch_size=20)
#         plt.scatter(x2.numpy(),y2.numpy())
#
#         plt.xlim(0,20)
#         plt.ylim(0,41)
#         plt.show()
#         plt.pause(0.5)
#
#     print(w.squeeze(),b.squeeze())



import torch as t
from torch.autograd import Variable
import matplotlib.pyplot  as plt

t.manual_seed(400)


def get_fake_data(batch_size=8):
    """
    产生随机数据：ｙ＝x*2+3,加上一些噪声
    :param batch_size:
    :return:
    """
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1)) * 3
    return x, y

w=Variable(t.rand(1,1),requires_grad=True)
b=Variable(t.zeros(1,1),requires_grad=True)

lr=0.001

for i in range(8000):
    x,y=get_fake_data()
    x,y=Variable(x),Variable(y)


    y_pred=x.mm(w)+b.expand_as(y)
    loss=0.5*(y_pred-y)**2
    loss=loss.sum()

    loss.backward()

    w.data.sub_(lr*w.grad.data)
    b.data.sub_(lr*b.grad.data)


    w.grad.data.zero_()
    b.grad.data.zero_()


    if i%1000==0:
        x=t.arange(0,20).view(-1,1).float()
        y=x.mm(w.data)+b.data.expand_as(x)
        plt.plot(x.numpy(),y.numpy())

        x2,y2=get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(),y2.numpy())

        plt.xlim(0,20)
        plt.ylim(0,41)
        plt.show()
        plt.pause(0.5)


print(w.data.squeeze(),b.data.squeeze())