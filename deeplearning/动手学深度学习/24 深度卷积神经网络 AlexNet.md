# 24 深度卷积神经网络 AlexNet

## 深度学习之前

### 核方法

2000年之前最火的机器学习模型：核方法

Learning with Kernels

- 特征提取
- 选择核函数来计算相关性
- 通过核函数计算后变成凸优化问题
- 由于凸优化因此有漂亮的定理

### 几何学

2000年计算机视觉关心的是从几何学过来的

Multiple View Geometry in computer vision

- 抽取特征
- 讲计算机视觉问题描述成几何问题（例如多相机）
- 建立（非）凸优化目标函数
- 漂亮的定理
- 如果假设满足了，效果非常好

### 特征工程

10～15年前，计算机视觉中最重要的是特征工程

- 关键是如何从原始图片中抽取特征

- 特征描述子：SIFT，SURF
- 视觉词袋（聚类）
- 最后用SVM

### Hardware

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 17.46.44.png" alt="截屏2021-12-12 17.46.44" style="zoom:50%;" />

## ImageNet（2010） VS. MNIST

|  图片  | 自然物体的彩色图片 | 手写数字的黑白图片 |
| :----: | :----------------: | :----------------: |
|  大小  |      469*387       |       28*28        |
| 样本数 |        1.2M        |        60K         |
|  类数  |        1000        |         10         |

## AlexNet

AlexNet赢得了2012年ImageNet竞赛

本质上是更深更大的LeNet

主要改进：

- 丢弃法
- 由Sigmoid变为ReLU激活函数
- 采用MaxPooling，使得输出值变大，梯度相对来说更大，训练更加容易

计算机视觉方法论的改变（变为端到端）

- 由人工特征提取（主要关心点）后到SVM变为通过CNN学习特征到Softmax回归

  - 构造CNN相对来说简单，比较容易跨到不同问题与学科

  - 因为特征与Softmax在网络里其实是一起训练的，从模型角度来讲其实就是一个，因此更加高效

## AlexNet架构

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.11.27.png" alt="截屏2021-12-12 21.11.27" style="zoom:50%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.14.57.png" alt="截屏2021-12-12 21.14.57" style="zoom:50%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.15.41.png" alt="截屏2021-12-12 21.15.41" style="zoom:50%;" />

## 更多细节

激活函数从Sigmoid变成了ReLU（减缓梯度消失）

全连接层的隐藏层后加入了丢弃层，进行正则化

==数据增强==（随机裁剪、改变亮度、色温）

- 因为卷积对位置很敏感，因此在训练时就模拟变化，增加大量变种，使得位置不敏感

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.19.01.png" alt="截屏2021-12-12 21.19.01" style="zoom:50%;" />

## 复杂度

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.21.26.png" alt="截屏2021-12-12 21.21.26" style="zoom:50%;" />

## 代码实现

```python
import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里，我们使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))
```

```python
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
    
# Conv2d output shape:	   torch.Size([1, 96, 54, 54])
# ReLU output shape:	     torch.Size([1, 96, 54, 54])
# MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
# Conv2d output shape:	   torch.Size([1, 256, 26, 26])
# ReLU output shape:	     torch.Size([1, 256, 26, 26])
# MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
# Conv2d output shape:	   torch.Size([1, 384, 12, 12])
# ReLU output shape:	     torch.Size([1, 384, 12, 12])
# Conv2d output shape:	   torch.Size([1, 384, 12, 12])
# ReLU output shape:	     torch.Size([1, 384, 12, 12])
# Conv2d output shape:	   torch.Size([1, 256, 12, 12])
# ReLU output shape:	     torch.Size([1, 256, 12, 12])
# MaxPool2d output shape:	 torch.Size([1, 256, 5, 5])
# Flatten output shape:	   torch.Size([1, 6400])
# Linear output shape:	   torch.Size([1, 4096])
# ReLU output shape:	     torch.Size([1, 4096])
# Dropout output shape:	   torch.Size([1, 4096])
# Linear output shape:	   torch.Size([1, 4096])
# ReLU output shape:	     torch.Size([1, 4096])
# Dropout output shape:	   torch.Size([1, 4096])
# Linear output shape:	   torch.Size([1, 10])
```

```python
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

```python
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 21.34.08.png" alt="截屏2021-12-12 21.34.08" style="zoom:50%;" />

