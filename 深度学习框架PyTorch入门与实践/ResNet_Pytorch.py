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
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    实现子module:Residual BLock
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):

        super(ResidualBlock, self).__init__()
        # 或者
        # nn.Module.__init__(self)

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, input):
        out = self.left(input)
        residual = input if self.right is None else self.right(input)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """
    实现主module：ResNet34

    """

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()

        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer，包含多个residual block
        :param inchannel:
        :param outchannel:
        :param block_num:
        :param stride:
        :return:
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, input):
        input = self.pre(input)

        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)

        input = F.avg_pool2d(input, 7)
        input = input.view(input.size(0), -1)
        return self.fc(input)


def main():
    model = ResNet()
    input = t.autograd.Variable(t.randn(1, 3, 224, 224))
    out = model(input)
    print(out)
    from torchvision import models
    resnet34=models.resnet34(pretrained=False,num_classes=1000)
    out1=resnet34(input)
    print(out1)



if __name__ == "__main__":
    main()
