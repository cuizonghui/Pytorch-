# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
from torchvision.models import squeezenet1_1
from models.basic_module import BasicModule
from torch import nn
from torch.optim import Adam


class SqueezeNet(BasicModule):
    def __init__(self, num_classes=2):
        super(SqueezeNet, self).__init__()
        self.model_name = 'squeezenet'
        self.model = squeezenet1_1(pretrained=True)

        # 修改原始的num_classes，预训练模型是１０００f分类
        self.model.num_classes = num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

    def forward(self, input):
        return self.model(input)

    def get_optimizer(self, lr, weight_decay):
        return Adam(self.model.classifier.parameters(), lr, weight_decay=weight_decay)
