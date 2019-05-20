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
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from transformer_net import TransformerNet
import utils
from PackedVGG import Vgg16
from torch.nn import functional as F
import tqdm
import os
import ipdb

mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]

class Config(object):
    image_size=256
    batch_size=8
    data_root='data/'
    num_workers=4
    use_gpu=True

    style_path=''
    lr=1e-3

    env='neural-style'
    plot_every=10
    gpu=True
    epoches=2

    content_weight=1e5
    style_weight=1e10

    model_path=None
    debug_file='/tmp/debugnn'

    content_path='input.png'
    result_path='output.png'


def train(**kwargs):
    opt=Config()
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)

    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis=utils.Viseulizer(opt.env)

    #数据加载
    transforms=tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x:x*255)
    ])
    dataset=tv.datasets.ImageFolder(opt.data_root,transforms)
    dataloader=data.DataLoader(dataset,opt.batch_size)

    #装换网络
    transformer=TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path,map_location=lambda _s,_:_s))
    transformer.to(device)

    #损失网络Vgg16
    vgg=Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad=False

    optimizer=t.optim.Adam(transformer.parameters(),opt.lr)

    #获取风格图片的数据
    style=utils.get_style_data(opt.style_path)
    vis.img('style',(style.data[0]*0.225+0.45).clamp(0,1))
    style=style.to(device)


    #风格图片的gram矩阵
    with t.no_grad():
        features_style=vgg(style)
        gram_style=[utils.gram_matrix(y) for y in features_style]

    style_meter=tnt.meter.AverageValueMeter()
    content_meter=tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for i ,(x,_) in tqdm.tqdm(enumerate(dataloader)):
            #训练
            optimizer.zero_grad()
            x=x.to(device)
            y=transformer(x)
            y=utils.normalize_batch(y)
            x=utils.normalize_batch(x)
            features_y=vgg(y)
            features_x=vgg(x)


            content_loss=opt.content_weight
            
            style_loss=0.
            for ft_y,gm_s in zip(features_y,gram_style):
                gram_y=utils.gram_matrix(ft_y)
                style_loss+=F.mse_loss(gram_y,gm_s.expand_as(gram_y))
            style_loss*=opt.style_weight
            
            total_loss=content_loss+ style_loss
            total_loss.backward()
            optimizer.step()
            
            #损失平滑
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())
            
            if (i+1)%opt.plot_every==0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                
                vis.plot('conten_loss',content_meter.value()[0])
                vis.plot('style_loss',style_meter.value()[0])
                vis.img('output',(y.data.cpu()[0]*0.225+0.45).clamp(min=0,max=1))
                vis.img('input',(x.data.cpu()[0]*0.225+0.45).clamp(min=0,max=1))
        vis.save([opt.env])
        t.save(transformer.state_dict(),'checkpoints/%s_style.pth'%epoch)
        

def stylize(**kwargs):
    opt=Config()
    
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')

    content_image=tv.datasets.folder.default_loader(opt.content_path)
    content_transform=tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x:x.mul(255))
        ])
    
    content_image=content_transform(content_image)
    content_image=content_image.unsqueeze(0).to(device).detach()

    style_model=TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path,map_location=lambda _s,_:_s))


    output=style_model(content_image)
    output_data=output.cpu().data[0]
    tv.utils.save_image((output_data/225).clamp(min=0,max=1),opt.result_path)
if __name__=='__main__':
    import fire
    fire.Fire()
