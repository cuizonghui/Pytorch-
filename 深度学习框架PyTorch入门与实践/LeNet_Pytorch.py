# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch as t

from torch.autograd import Variable
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(LeNet, self).__init__()

        # 卷积函数中的参数１代表输入图片的通道数为１，６表示输出通道数为６，５表示卷积核的大小
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全连接层y=Wx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积－＞激活－＞池化
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    import numpy as np
    lenet = LeNet()
    # print(lenet)
    # # lenet.forward(t.ones((1,28,28,1)))
    # # print(list(lenet.parameters()))
    #
    # params=list(lenet.parameters())
    #
    # for name,parameters in lenet.named_parameters():
    #     print(name,':',parameters.size())

    input=Variable(t.randn(1,1,32,32))
    output=lenet(input)
    target=Variable(t.arange(0,10)).float()
    criterion=nn.MSELoss()
    # loss=criterion(output,target)
    # print(loss)
    #
    # lenet.zero_grad()
    # loss.backward()
    #
    # learning_rate=0.01
    optimizer=optim.SGD(lenet.parameters(),lr=0.01)
    # for f in lenet.parameters():
    #     f.data.sub_(f.grad.data*learning_rate) #inplace减法

    optimizer.zero_grad()

    #计算损失
    output=lenet(input)
    loss=criterion(output,target)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
