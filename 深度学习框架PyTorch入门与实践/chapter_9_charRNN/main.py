# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import sys,os
import torch as t
from data import get_data
from model import PoetryModel
from torch import nn
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb


class Config(object):
    data_path='data/'##诗歌的文本文件存放路径
    pickle_path='tang.npz'
    author=None
    constrain=None
    category='poet.tang'
    lr=1e-3
    weight_decay=1e-4
    use_gpu=True
    epoch=20
    batch_size=128
    maxlen=125
    plot_every=20
    env='poetry'
    max_gen_len=200 ##生成诗歌最大长度
    debug_file='/tmp/debugp'
    model_path=None
    prefix_words='细雨鱼儿出，微风燕子斜'
    start_words='闲云潭影日悠悠'
    acrostic=False #是否是藏头诗
    model_prefx='checkpoints/tang'

opt=Config()

def generate(model,start_words,ix2word,word2ix,prefix_words=None):
    '''
    给定几个词，根据这几个词连接生成一首完整的诗歌
    :param model:
    :param start_words:
    :param ix2word:
    :param word2ix:
    :param prefix_words:
    :return:
    '''

    results=list(start_words)
    start_word_len=len(start_words)

    input=t.Tensor([word2ix['<START>']]).view(1,1).long()
    if opt.use_gpu:input=input.cuda()
    hidden=None

    if prefix_words:
        for word in prefix_words:
            output,hidden=model(input,hidden)
            input=input.data.new([word2ix[word]]).view(1,1)

    for i in range(opt.max_gen_len):
        output,hidden=model(input,hidden)

        if i <start_word_len:
            w=results[i]
            input=input.data.new([word2ix[w]]).view(1,1)
        else:
            top_index=output.data[0].topk(1)[1][0].item()
            w=ix2word[top_index]
            results.append(w)
            input=input.data.new([top_index]).view(1,1)
        if w=='<EOP>':
            del results[-1]
            break
    return results


def gen_acrostic(model,start_words,ix2word,word2ix,prefix_words=None):
    '''
    生成藏头诗
    :param model:
    :param start_words:
    :param ix2word:
    :param word2ix:
    :param prefix_word:
    :return:
    '''
    results=[]
    start_word_len=len(start_words)
    input=(t.Tensor([word2ix['START']]).view(1,1).long())
    if opt.use_gpu:input=input.cuda()
    hidden=None

    index=0
    pre_word='<START>'

    if prefix_words:
        for word in prefix_words:
            output,hidden=model(input,hidden)
            input=(input.data.new([word2ix[word]])).view(1,1)
    for i in range(opt.max_gen_len):
        output,hidden=model(input,hidden)
        top_index=output.data[0].topk(1)[1][0].item()
        w=ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}):
            if index ==start_word_len:
                break
            else:
                w=start_words[index]
                index+=1
                input=(input.data.new([word2ix[w]])).view(1,1)
        else:
            input=(input.data.new([word2ix[w]])).view(1,1)
        results.append(w)
        pre_word=w
    return results

def train(**kwargs):
    for k,v in kwargs.items():
        setattr(opt,k,v)
    opt.device=t.device('cuda') if opt.use_gpu else t.device('cpu')

    device=opt.device
    vis=Visualizer(env=opt.env)

    data,word2ix,ix2word=get_data(opt)
    data=t.from_numpy(data)
    dataloader=t.utils.data.DataLoader(data,opt.batch_size,shuffle=True,num_workers=1)


    #模型定义
    model=PoetryModel(len(word2ix),128,256)
    optimizer=t.optim.Adam(model.parameters(),lr=opt.lr)
    criterion=nn.CrossEntropyLoss()
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(device)

    loss_meter=meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for i,data_ in tqdm.tqdm(enumerate(dataloader)):
            #训练
            data_=data_.long().transpose(1,0).contiguous()
            data_=data_.to(device)
            optimizer.zero_grad()
            input_,target=data_[:-1,:],data_[1:,:]
            output,_=model(input_)
            loss=criterion(output,target.view(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            if os.path.exists(opt.debug_file):
                ipdb.set_trace()

            vis.plot('loss', loss_meter.value()[0])

            # 诗歌原文
            poetrys = [[ix2word[_word] for _word in data_[:, _iii].tolist()]
                       for _iii in range(data_.shape[1])][:16]
            vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')

            gen_poetries = []
            # 分别以这几个字作为诗歌的第一个字，生成8首诗
            for word in list(u'春江花月夜凉如水'):
                gen_poetry = ''.join(generate(model, word, ix2word, word2ix))
                gen_poetries.append(gen_poetry)
            vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')

    t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))

def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的诗
    """

    for k, v in kwargs.items():
        setattr(opt, k, v)
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), 128, 256);
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    gen_poetry = gen_acrostic if opt.acrostic else generate
    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))

if __name__ == '__main__':
    import fire

    fire.Fire()
