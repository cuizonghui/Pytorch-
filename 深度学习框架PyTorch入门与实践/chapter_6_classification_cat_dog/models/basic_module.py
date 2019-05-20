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
import time

class BasicModule(t.nn.Module):
    """
    封装nn.MOdule，主要是提供save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))

    def load(self,path):
        """
        可加载模型的路径
        :param path:
        :return:
        """
        self.load_state_dict(t.load(path))

    def save(self,name=None):
        """
        保存模型，默认使用＇模型名字＋时间＇作为文件名
        :param name:
        :return:
        """
        if name is None:
            prefix='checkpoints/'+self.model_name+'_'
            name=time.strftime(prefix+'%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(),name)
        return name