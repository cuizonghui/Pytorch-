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
from torch import nn
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(Linear,self).__init__()
        self.w=nn.Parameter(t.randn(in_features,out_features))

        self.b=nn.Parameter(t.randn(out_features))

    def forward(self, input):
        input=input.mm(self.w)
        return input+self.b.expand_as(input)


def main():
    layer=Linear(4,3)
    input=Variable(t.randn(2,4))
    output=layer(input)
    print(output)


    for name,parameter in layer.named_parameters():
        print(name,parameter)


if __name__=='__main__':
    main()