# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
from itertools import chain
import visdom
import torch
import time
import torchvision as tv
import numpy as np


class Visualizer():
    '''
    封装了visdom的基本操作，但你仍然可以通过self.vis.function，调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        :param env:
        :param kwargs:
        :return:
        '''
        self.vis = visdom.Visdom(env, use_incoming_socket=False, **kwargs)
        return self

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=(name), opts=dict(title=name),
                      update=None if x == 0 else 'append')

        self.index[name] = x + 1

    def img(self, name, img_):
        if len(img_._size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(), win=(name), opts=dict(title=name))

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        self.img(name, tv.utils.make_grid(input_3d.cup()[0].unsqueeze(1).clamp(
            max=1, min=0
        )))

    def log(self, info, win='log_text'):
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        ))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, item):
        return getattr(self.vis, item)
