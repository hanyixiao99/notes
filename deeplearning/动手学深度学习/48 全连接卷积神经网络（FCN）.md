# 48 全连接卷积神经网络（FCN）

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 11.49.46.png" alt="截屏2021-12-16 11.49.46" style="zoom:50%;" />

FCN使用深度神经网络来做语义分割的奠基性工作

它用转置卷积层来替换CNN最后的全连接层，从而可以实现每个像素的预测

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 11.51.40.png" alt="截屏2021-12-16 11.51.40" style="zoom:50%;" />

对每个像素类别的预测存在通道K中，K=类别数

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 11.54.43.png" alt="截屏2021-12-16 11.54.43" style="zoom:50%;" />

## 代码实现

```python
%matplotlib inline
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

使用在ImageNet数据集上预训练的ResNet-18模型来提取图像特征

```python
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
# [Sequential(
#    (0): BasicBlock(
#      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#      (relu): ReLU(inplace=True)
#      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#      (downsample): Sequential(
#        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#      )
#    )
#    (1): BasicBlock(
#      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#      (relu): ReLU(inplace=True)
#      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#    )
#  ),
#  全局平均池化层
#  AdaptiveAvgPool2d(output_size=(1, 1)),
#  最后一层线性输出层
#  Linear(in_features=512, out_features=1000, bias=True)]
```

创建一个全卷积网络实例`net`，将上述模型最后两层去掉

(从ResNet模型中获得抽取特征模块)

```python
net = nn.Sequential(*list(pretrained_net.children())[:-2])

X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
# torch.Size([1, 512, 10, 15])
```

使用1x1卷积层将输出通道数转换为Pascal VOC2012数据集的类数（21类）

将特征图的高度和宽度增加32倍

```python
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

pass 