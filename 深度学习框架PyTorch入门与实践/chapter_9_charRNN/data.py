# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import sys
import  os
import json
import re
import numpy as np

def  _parseRawData(author=None,constrain=None,src='./chinese-poetry/json/simplified',
                   category='poet.tang'):
    """
    处理json文件，返回诗歌内容
    :param author: 作者姓名
    :param constrain: 长度限制
    :param sr: 文件存放路径
    :param category: 类别
    :return: list
    """

    def sentenceParse(para):
        result,number=re.subn(u"(.*)","",para)
        result,number=re.subn(u"{.*}","",result)
        result,number=re.subn(u"《.*》","",result)
        result,number=re.subn(u"《.*》","",result)
        result,number=re.subn(u"[\]\[]","",result)
        r=""
        for s in result:
            if s not in set('0123456789-'):
                r+=s
        r,number=re.subn(u"。。", u"。", r)
        return  r


    def handleJson(file):
        rst=[]
        data=json.loads(open(file).read())
        for poetry in data:
            pdata=""
            if (author is not None and poetry.get('author')!=author):
                continue
            p=poetry.get('paragraphs')
            flag=False
            for s in p:
                sp=re.split(u"[,!。]",s)
                for tr in sp:
                    if constrain is not None and len(tr)!=constrain and len(tr)!=0:
                        flag=True
                    if flag:
                        break
                if flag:
                    continue
                for sentence in poetry.get("paragraphs"):
                    pdata+=sentence
                pdata=sentenceParse(pdata)
                if pdata!="":
                    rst.append(pdata)
        return  rst

    data=[]
    for filename in os.listdir(src):
        if filename.startswith(category):
            data.extend(handleJson(src+filename))
    return data


def pad_sequences(sequences,maxlen=None,dtype='int32',padding='pre',
                  truncating='pre',value=0):
    if  not hasattr(sequences,'__len__'):
        raise ValueError('sequences must be iterables')

    lengths=[]
    for x in sequences:
        if not hasattr(x,'__len__'):
            raise ValueError('sequences must be a lsit of iterables. Found non-iterable:'+str(x))
        lengths.append(len(x))
    num_samples=len(sequences)
    if maxlen is None:
        maxlen=np.max(lengths)

    sample_shape=tuple()
    for s in sequences:
        if len(s)>0:
            sample_shape=np.asarray(s).shape[1]
            break
    x=(np.ones((num_samples,maxlen)+sample_shape)*value).astype(dtype)

    for idx,s in enumerate(sequences):
        if not len(s):
            continue
        elif truncating=='pre':
            truc=s[-maxlen:]
        else:
            raise ValueError('Truncating type "%s" not understood'%truncating)

        truc=np.asarray(truc,dtype)
        if truc.shape[1:]!=sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different'
                             'from expected shape %s'%(truc.shape[1:],idx,sample_shape))

        if padding=='post':
            x[idx,:len(truc)]=truc
        elif padding=='pre':
            x[idx,-len(truc)]=truc
        else:
            raise ValueError('Padding type "%s" not understood'%padding)
    return x


def get_data(opt):

    if os.path.exists(opt.pickle_path):
        data=np.load(opt.pickle_path)
        data,word2ix,ix2word=data['data'],data['word2ix'].item(),data['ix2word'].item()
        return data,word2ix,ix2word

    data=_parseRawData(opt.author,opt.constrain,opt.data_path,opt.category)
    words={_word for _sentence in data for _word in _sentence}
    word2ix={_word:_ix for _ix,_word in enumerate(words)}
    word2ix['<EOP>']=len(word2ix)
    word2ix['<START>']=len(word2ix)
    word2ix['</s>']=len(word2ix)
    ix2word={_ix:_word for _word,_ix in list(word2ix.items())}

    ##为每首诗歌增加起始符和终止符
    for i in range(len(data)):
        data[i]=["<START>"]+list(data[i])+["<EOP>"]

    new_data=[[word2ix[_word] for _word in _sentence] for _sentence in data]

    pad_data=pad_sequences(new_data,maxlen=opt.maxlen,padding='pre',truncating='post',
                           value=len(word2ix)-1)

    #保存成二级制
    np.savez_compressed(opt.pickle_path,pad_data,word2ix,ix2word)
    return pad_data,word2ix,ix2word
