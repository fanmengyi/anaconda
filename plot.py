from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt


train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)
train_loader = Data.DataLoader(dataset=train_data,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0)
#获取batch数据
for step, (b_x,b_y) in enumerate(train_loader):  # x是数据，y是标签
    if step>0:
        break
batch_x=b_x.squeeze().numpy()  #将四维张量移除第一维(通道的维度去掉)，并转为numpy数据
batch_y=b_y.numpy()  #将张量转为numpy数据
class_lable=train_data.classes  #训练集标签
print(class_lable)

#可视化一个Batch的图像
plt.figure(figsize=(12,5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4,16,ii+1)
    plt.imshow(batch_x[ii,:,:], cmap=plt.cm.gray)
    plt.title(class_lable[batch_y[ii]],size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()