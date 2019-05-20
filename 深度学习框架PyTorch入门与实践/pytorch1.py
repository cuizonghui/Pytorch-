# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import torch as t
print(t.__version__)
import numpy as np
from torch.autograd import Variable
x=Variable(t.ones(2,2),requires_grad=True)
print(x)
y=x.sum()
print(y)
print(y.grad_fn)
print(y.data)
print(y.grad)

y.backward()
print(x.data)
print(x.grad)
print(x.grad_fn)

y.backward()
print(x.grad)

y.backward()
print(x.grad)

x.grad.data.zero_()

y.backward()
print(x.grad)


x=Variable(t.ones(4,5))
print(x)
y=t.cos(x)
print(x)
print(y)




# a=t.ones(5)
# print(a)
#
# b=a.numpy()
# print(b)
#
# a=np.ones(10)
# b=t.from_numpy(a)
# print(a)
# print(b)
#
# b.add_(1)
# print(a)
# print(b)
#
# x=t.Tensor(5,3)
# print(x)
#
# x=t.rand(5,3)
# print(x)
#
# print(x.size())
#
# y=t.rand(5,3)
# print(y)
# if t.cuda.is_available():
#     x=x.cuda()
#     y=y.cuda()
#     print(x+y)



#
# print(x+y)
#
# print(t.add(x,y))
#
# result=t.Tensor(5,3)
# t.add(x,y,out=result)
#
# print(result)
# print('原始y')
# print(y)
#
# print('第一种加法，ｙ的结果')
# y.add(x)
# print(y)
#
#
# print('第二种加法，ｙ的结果')
# y.add_(x)
# print(y)
#
# print(x[:,1])