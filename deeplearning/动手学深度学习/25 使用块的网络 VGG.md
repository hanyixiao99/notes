# 25 使用块的网络 VGG

<!--使用可重复使用的卷积块来构建深度卷积神经网络-->

<!--不同的卷积块个数和超参数可以得到不同复杂度的变种-->

## VGG（2013）

AlexNet比LeNet通过采用更深更大的网络来得到更好的精度，能不能更深和更大

选项：

- 更多的全连接层（太贵）
- 更多的卷积层
- ==将卷积层组合成块==

## VGG块

AlexNet思路的拓展

深 vs. 宽

- 5x5卷积（计算量大）
- 3x3卷积
- 深但窄效果更好

VGG块

- 3x3卷积（填充1）（n层，m通道）
- 2x2最大池化层（步幅2）

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.41.43.png" alt="截屏2021-12-12 21.41.43" style="zoom:50%;" />

## VGG架构

多个VGG块后接全连接层

不同次数的重复块得到不同的架构（VGG-16，VGG-19）

## 进度

LeNet（1995）

- 2卷积+池化层
- 2全连接层

AlexNet

- 更大更深
- ReLu，Dropout，数据增强

VGG

- 更大更深的AlexNet（重复的VGG块）

## GluonCv Model Zoo

**https://cv.gluon.ai/model_zoo/classification.html**

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.47.18.png" alt="截屏2021-12-12 21.47.18" style="zoom:50%;" />

## 代码实现

VGG块

```python
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

VGG网络

```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

观察每个层的输出形状

```python
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
    
# Sequential output shape:         torch.Size([1, 64, 112, 112])
# Sequential output shape:         torch.Size([1, 128, 56, 56])
# Sequential output shape:         torch.Size([1, 256, 28, 28])
# Sequential output shape:         torch.Size([1, 512, 14, 14])
# Sequential output shape:         torch.Size([1, 512, 7, 7])
# Flatten output shape:        torch.Size([1, 25088])
# Linear output shape:         torch.Size([1, 4096])
# ReLU output shape:           torch.Size([1, 4096])
# Dropout output shape:        torch.Size([1, 4096])
# Linear output shape:         torch.Size([1, 4096])
# ReLU output shape:           torch.Size([1, 4096])
# Dropout output shape:        torch.Size([1, 4096])
# Linear output shape:         torch.Size([1, 10])
```

由于VGG-11比AlexNet计算量更大，因此构建一个通道数较少的网络

```python
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

模型训练

```python
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# loss 0.179, train acc 0.933, test acc 0.918
# 2550.1 examples/sec on cuda:0
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.58.29.png" alt="截屏2021-12-12 21.58.29" style="zoom:50%;" />