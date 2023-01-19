# 26 网络中的网络 NiN

<!--NiN块使用卷积层加两个1x1卷积层，后者对每个像素增加了非线性-->

<!--NiN使用全局平均池化层来替代VGG和AlexNet中的全连接层-->

<!--不容易过拟合，更少的参数个数-->

## 全连接层的问题

卷积层需要较少的参数 ci * co * k^2

但卷积层后的第一个全连接层的参数

- LeNet 16 * 5 * 5 * 120 = 48k
- AlexNet 256 * 5 * 5 * 4096 = 26M
- VGG 512 * 7 * 7 * 4096 = 102M

## NiN块

一个卷积层后跟两个卷积层（全连接）

- 步幅为1，无填充，输出形状跟卷积层输出一样
- 起到全连接层的作用

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 22.06.41.png" alt="截屏2021-12-12 22.06.41" style="zoom:50%;" />

## NiN架构

无全连接层

交替使用NiN块和步幅为2的最大池化层

- 逐步减小高宽和增大通道数

最后使用全局平均池化层得到输出

- 其输入通道数是类别数

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 22.09.45.png" alt="截屏2021-12-12 22.09.45" style="zoom:50%;" />

## 代码实现

NiN块

```python
import torch
from torch import nn
from d2l import torch as d2l

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

NiN模型

```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
```

