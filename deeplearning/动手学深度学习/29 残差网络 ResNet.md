# 29 残差网络 ResNet

<!--残差块使得很深的网络更加容易训练（甚至可以训练一千层网络）-->

<!--残差网络对随后的深层神经网络设计产生了深远影响-->

<!--无论是卷积类网络还是全连接类网络-->

 加更多的层总是改进精度吗？——ResNet核心思想：加更多的层至少不会变差

## 残差块

串联一个层改变函数类，我们希望能扩大函数类

残差块加入快速通道（右边）来得到fx=x+gx的结构

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-13 16.47.29.png" alt="截屏2021-12-13 16.47.29" style="zoom:50%;" />

## 残差块细节

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-13 16.50.31.png" alt="截屏2021-12-13 16.50.31" style="zoom:50%;" />

## 不同的残差块

可以加在不同位置

## ResNet块

高宽减半残差块（步幅2）

后接多个高宽不变的残差块

## ResNet架构

类似VGG和GoogLeNet的总体架构

但替换成了ResNet块

## 代码实现

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

## ResNet为什么能训练1000层的模型

首先假设有一个网络 y = fx，里面有一个需要更新的权重w

