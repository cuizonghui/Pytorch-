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
import ipdb
import torch as t
import torchvision as tv
import tqdm
from model import NetG, NetD
from torchnet.meter import AverageValueMeter


class Config(object):

    data_path = 'data/'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    image_size = 96  # 图片尺寸
    batch_size = 256
    max_epoch = 200
    lr1 = 2e-4  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    beta1 = 0.5  # Adam优化器的beta１参数
    use_gpu = True
    nz = 100  # 噪声维度
    ngf = 64  # 生成器feature map数
    ndf = 64  # 　判别器feature map数

    save_path = 'imgs/'

    vis = True  # 是否使用visdom可视化
    env = 'GAN'  # visdom的env
    plot_every = 20  # 每隔２０个ｂａｔｃｈ,visdom画图一次

    debug_file = '/tmp/debuggan'
    d_every = 1  # 每个batch训练一次判别器
    g_every = 5  # 每５个ｂａｔｃｈ训练一次生成器
    save_every = 10  # 每１０个epoch保存一次模型

    netd_path = 'checkpoints/netd_199.pth'
    netg_path = 'checkpoints/netg_199.pth'

    # 测试时用的参数
    gen_img = 'result.png'
    # 从５１２张生成的图片中保存最好的６４张
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1
    gpu=True

opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.gpu else t.device('cpu')
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = tv.datasets.ImageFolder(opt.data_path, transform=transforms)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.num_workers,
                                         drop_last=True)

    # 网络
    netg, netd = NetG(opt), NetD(opt)

    def map_location(storage, loc): return storage
    if opt.netd_path:
        netd.load_state_dict(t.load(opt.netd_path, map_location=map_location))
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))

    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(), opt.lr1, betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(), opt.lr2, betas=(opt.beta1, 0.999))
    criterion=t.nn.BCELoss().to(device)

    #真图片标签为１，加图片标签为０
    true_labels=t.ones(opt.batch_size).to(device)
    fake_labels=t.zeros(opt.batch_size).to(device)
    fix_noises=t.randn(opt.batch_size,opt.nz,1,1).to(device)
    noises=t.randn(opt.batch_size,opt.nz,1,1).to(device)


    errorg_meter=AverageValueMeter()
    errord_meter=AverageValueMeter()

    epochs=range(opt.max_epoch)
    for epoch in iter(epochs):
        for i,(img,_) in tqdm.tqdm(enumerate(dataloader)):
            real_img=img.to(device)

            if i%opt.d_every==0:
                optimizer_d.zero_grad()
                output=netd(real_img)
                error_d_real=criterion(output,true_labels)
                error_d_real.backward()

                #尽可能把假图片判别为错误
                noises.data.copy_(t.randn(opt.batch_size,opt.nz,1,1))
                fake_img=netg(noises).detach()
                output=netd(fake_img)##根据噪声生成假图
                error_d_fake=criterion(output,fake_labels)
                error_d_fake.backward()
                optimizer_d.step()

                error_d=error_d_fake+error_d_real
                errord_meter.add(error_d.item())


            if i%opt.g_every==0:
                #训练生成器
                optimizer_g.zero_grad()
                noises.data.copy_(t.randn(opt.batch_size,opt.nz,1,1))
                fake_img=netg(noises)
                output=netd(fake_img)
                error_g=criterion(output,true_labels)
                error_g.backward()
                optimizer_g.step()
                errorg_meter.add(error_g.item())

            if opt.vis and i%opt.plot_every==opt.plot_every-1:
                ##可视化
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                fix_fake_imgs=netg(fix_noises)
                vis.images(fix_fake_imgs.detach().cpu().numpy()[:64]*0.5+0.5,win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:64]*0.5+0.5,win='real')
                vis.plot('errord',errord_meter.value()[0])
                vis.plot('errorg',errorg_meter.value()[0])
            if (epoch+1)%opt.save_every==0:
                tv.utils.save_image(fix_fake_imgs.data[:64],'%s/%s.png'%(
                    opt.save_path,epoch),normalize=True,range=(-1,1))

                t.save(netd.state_dict(),'checkpoints/netd_%s.pth'%epoch)
                t.save(netg.state_dict(),'checkpoints/netg_%s.pth'%epoch)
                errord_meter.reset()
                errorg_meter.reset()


def generate(**kwargs):
    """
    随机生成动漫图像，并根据netd的分数选择较好的
    :param kwargs:
    :return:
    """
    for k_,v_ in kwargs.items():
        setattr(opt,k_,v_)

    device=t.device('cuda' if opt.gpu else t.device('cpu'))

    netg,netd=NetG(opt).eval(),NetD(opt).eval()
    noises=t.randn(opt.gen_search_num,opt.nz,1,1).normal_(opt.gen_mean,opt.gen_std)
    noises=noises.to(device)

    map_location=lambda storage,loc:storage
    netd.load_state_dict(t.load(opt.netd_path,map_location))
    netg.load_state_dict(t.load(opt.netg_path,map_location))
    netd.to(device)
    netg.to(device)

    fake_img=netg(noises)
    scores=netd(fake_img).detach()

    indexs=scores.topk(opt.gen_num)[1]
    result=[]
    for i in indexs:
        result.append(fake_img.data[i])
    tv.utils.save_image(t.stack(result),opt.gen_img,normalize=True,range=(-1,1))

if __name__=='__main__':
    import fire
    fire.Fire()

