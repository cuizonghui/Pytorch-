# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        目标，获取所有图片的地址，并根据训练，验证，测试划分数据
        :param root:
        :param transforms:
        :param train:
        :param test:
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:

            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分训练验证,验证：训练＝３：７
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]
        if transforms is None:

            # 转换数据操作，测试验证和训练数据有所差别
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        返回一张图片数据,如果是测试集没有图片ｉｄ，如1000.jpg返回1000
        :param index:
        :return:
        """

        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        """
        返回数据集中所有图片的个数
        :return:
        """
        return len(self.imgs)


def main():

    train_dataset_root = './test'
    batch_size = 10
    train_dataset = DogCat(train_dataset_root, train=True, test=True)
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size)

    for i, (img, label) in enumerate(trainloader):
        print(i, label)


if __name__ == "__main__":
    main()
