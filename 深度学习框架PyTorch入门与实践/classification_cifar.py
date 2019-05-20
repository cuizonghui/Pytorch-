# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import torchvision as tv
import torch as t
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()  # 可以把Tensor转成Image,方便可视化
import matplotlib.pyplot as plt
import  torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import optim
import numpy as np
import cv2

class LeNet(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(LeNet, self).__init__()

        # 卷积函数中的参数１代表输入图片的通道数为１，６表示输出通道数为６，５表示卷积核的大小
        self.conv1 = nn.Conv2d(3, 6, 5)
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




transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                                ])

def convert_chw_to_hwc(input_data):
    c,h,w=input_data.shape
    output_data=np.zeros([h,w,c],dtype=np.uint8)
    for i in range(c):
        output_data[:,:,i]=input_data[i,:,:]
    return output_data



# 训练集
trainset = tv.datasets.CIFAR10(root='./data/',
                               train=True,
                               download=True,
                               transform=transform)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# 测试集
testset = tv.datasets.CIFAR10('./data/',
                              train=False,
                              download=True,
                              transform=transform)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data, label) = trainset[100]
print(classes[label])
data_np=np.uint8(255*data.numpy())
data_np=convert_chw_to_hwc(data_np)
data_np=cv2.resize(data_np,(100,100))

# data_np=np.reshape(data_np,(data_np.shape[1],data_np.shape[2],data_np.shape[0]))
# data_np=np.resize(data_np,new_shape=[100,100])
# plt.figure(1,figsize=(120,120))
# plt.imshow(data_np)
# plt.show()



# print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
# images_np=np.uint8(255*images.numpy())
# images_np=convert_chw_to_hwc(images_np)
# images_np=cv2.resize(images_np,(100,100))
# plt.figure(1,figsize=(120,120))
# plt.imshow(data_np)
# plt.show()

def main():
    epochs=20000
    # dataiter = iter(trainloader)
    if t.cuda.is_available():
        lenet = LeNet().cuda()
        # input=Variable(t.randn(1,1,32,32))
        # output=lenet(input)



        target=Variable(t.arange(0,10)).float()
        criterion=nn.CrossEntropyLoss()

        optimizer=optim.SGD(lenet.parameters(),lr=0.001,momentum=0.9)
        for epoch in range(epochs):
            running_loss=0.0
            for i,data in enumerate(trainloader,0):
                inputs,labels=data
                inputs=inputs.cuda()
                labels=labels.cuda()
                inputs,labels=Variable(inputs),Variable(labels)

                optimizer.zero_grad()

                outputs=lenet(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                if i%2000==1999:
                    print('[%d,%5d] loss:%.3f'%(epoch+1,i+1,running_loss/2000))
                    running_loss=0.0
        print('Finished Training')






if __name__ == "__main__":
    main()
