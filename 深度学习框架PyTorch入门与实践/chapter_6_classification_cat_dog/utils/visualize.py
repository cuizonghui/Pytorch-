# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import visdom
import time
import numpy as np


class Visualizer(object):
    '''
    封装visdom的基本操作，但仍然可以通过self.vis.funciton或者self.fucntion调用原生的接口
    例如:self.text('hello visdom')
    self.histogram(t.randn(1000))
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 保存（'loss',23）
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        :param env:
        :param kwargs:
        :return:
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        :param d:
        :return:
        '''
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss',1.00)
        :param name:
        :param y:
        :param kwargs:
        :return:
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=(name),
                      opts=dict(title=name), update=None if x == 0 else 'append',
                      **kwargs)

        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        :param name:
        :param img_:
        :param kwargs:
        :return:
        '''

        self.vis.images(img_.cpu().numpy(), win=(name), opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        :param info:
        :param win:
        :return:
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text,win)

    def __getattr__(self, item):
        return getattr(self.vis, item)
