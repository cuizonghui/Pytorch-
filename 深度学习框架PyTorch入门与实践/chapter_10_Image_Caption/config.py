# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------

class Config:
    caption_data_path='caption.pth'
    img_path='./caption_data/'
    img_feature_path='results.pth'
    scale_size=300
    img_size=224
    batch_size=8
    shuffle=True
    num_workers=4
    rnn_hidden=256
    embedding_dim=256
    num_layers=2
    share_embedding_weights=False
    prefix=''
    env='caption'
    plot_every=10
    debug_file='debugC'

    model_cpkt=None
    lr=1e-3
    use_gpu=True
    epoch=1

    test_img='img/example.jpeg'