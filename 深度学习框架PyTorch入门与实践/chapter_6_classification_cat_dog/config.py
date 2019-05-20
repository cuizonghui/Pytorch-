# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import warnings
import torch as t

class DefaultConfig(object):
    env='default'
    model='AlexNet'

    train_data_root='./data/train/'
    test_data_root='/data/test'
    load_model_path='checkpoints/model.pth'

    batch_size=128
    use_gpu=True
    num_workers=4
    print_freq=20

    debug_file='/tmp/debug'
    result_file='result.csv'

    max_epoch=10
    lr=0.1
    lr_decay=0.95
    weight_decay=1e-4

    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn('Warning:opt has not attribut %s'%k)
            setattr(self,k,v)
        opt.device=t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k,getattr(self,k))
opt=DefaultConfig()


