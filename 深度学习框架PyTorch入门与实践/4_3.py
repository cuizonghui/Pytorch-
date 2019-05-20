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
from torch import optim
import torch as t


from torch.autograd import Variable
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
    def forward(self, input):
        input=self.features(input)
        input=input.view(-1,16*5*5)
        input=self.classifier(input)
        return input

def main():
    net=Net()

    optimizer=optim.SGD(params=net.parameters(),lr=1)
    optimizer.zero_grad()

    input=Variable(t.randn(1,3,32,32))
    output=net(input)
    output.backward(output)

    optimizer.step()



if __name__=='__main__':
    main()
