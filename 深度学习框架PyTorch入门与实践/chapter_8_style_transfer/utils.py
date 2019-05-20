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
import numpy as np
import visdom
import time

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]


def gram_matrix(y):
    '''
    输入形状b,c,h,w
    输出形状b,c,c
    :param y:
    :return:
    '''

    (b,ch,h,w)=y.size()
    features=y.view(b,ch,w*h)
    features_t=features.transpose(1,2)
    gram=features.bmm(features_t)/(ch*h*w)
    return gram


def get_style_data(path):
    '''
    加载风格图片
    :param path:
    :return:
    '''
    style_transform=tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD),
    ])

    style_image=tv.datasets.folder.default_loader(path)
    style_tensor=style_transform(style_image)
    return style_tensor.unsqueeze(0)


def normalize_batch(batch):
    '''
    输入b,ch,h,w 0~255,是一个Variable
    输出b,ch,h,w 大约－２～２，是一个Variable
    :param batch:
    :return:
    '''
    mean=batch.data.new(IMAGENET_MEAN).view(1,-1,1,1)
    std=batch.data.new(IMAGENET_STD).view(1,-1,1,1)
    mean=t.autograd.Variable(mean.expand_as(batch.data))
    std=t.autograd.Variable(std.expand_as(batch.data))
    return (batch/255.0-mean)/std



class Visualizer():
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env,use_incoming_socket=False,  **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        """
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
