# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
from torchvision import datasets,transforms

from torch.utils.data import DataLoader
from torchvision.utils import make_grid,save_image

normalize=transforms.Normalize(mean=[0.4,0.4,0.4],std=[0.2,0.2,0.2])
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x:x.repeat(3,1,1)),
    normalize
])

# transform = transforms.Compose(
#     [ transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3,1,1)), normalize]) # 修改的位置



dataset=datasets.MNIST('data/',download=True,train=False,transform=transform)

dataloader=DataLoader(dataset,shuffle=True,batch_size=64)

dataiter=iter(dataloader)
img=make_grid(next(dataiter)[0],8)
to_pil=transforms.ToPILImage()

to_pil(img).show()

dd=0

from tensorboard_logger import Logger
logger=Logger(logdir='./log',flush_secs=2)

for i in range(100):
    logger.log_value('loss',10-i**0.5,step=i)
    logger.log_value('accuray',i**0.5/10,step=i)

dd=-0
