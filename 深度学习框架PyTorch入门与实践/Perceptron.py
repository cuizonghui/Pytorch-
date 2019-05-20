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
from Linear import Linear


class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Perceptron, self).__init__()  # 或者nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, input):
        input = self.layer1(input)
        input = t.sigmoid(input)
        return self.layer2(input)


def main():
    perceptron = Perceptron(3, 4, 1)
    # for name,parameter in perceptron.named_parameters():
    #     print(name,parameter.size(),parameter)

    for para in perceptron.parameters():
        print(para)


if __name__ == '__main__':
    main()
