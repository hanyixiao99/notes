# 16 PyTorch神经网络基础

## 层和块

首先回顾一下多层感知机 

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256,10))

X = torch.rand(2, 20)
net(X)
# tensor([[-0.1346, -0.0607,  0.0079, -0.1499, -0.0723,  0.0714,  0.1308, -0.1113,
#          -0.1019,  0.4291],
#         [-0.1361, -0.1472, -0.0497, -0.2400, -0.0750,  0.2048,  0.1763, -0.0087,
#          -0.0441,  0.3950]], grad_fn=<AddmmBackward0>)
```

`nn.Sequential`定义了一种特殊的`Module`

在PyTocrh中`Module`是一个很重要的概念，可以认为是说任何一个层和一个神经网络都应该是一个Module的子类

自定义块

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用`MLP`的父类`Module`的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数`params`（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入`X`返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

实例化多层感知机的层，然后在每次调用正向传播函数时调用这些层

```python
net = MLP()
net(X)
# tensor([[ 0.0950, -0.0575, -0.0460,  0.0683, -0.1133,  0.0240, -0.0984, -0.0503,
#          -0.0466, -0.0547],
#         [-0.0256, -0.0895, -0.1116,  0.1063, -0.0565,  0.1329, -0.2335,  0.0871,
#          -0.0917, -0.0876]], grad_fn=<AddmmBackward0>)
```

顺序块

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，`module`是`Module`子类的一个实例。我们把它保存在'Module'类的成员
            # 变量`_modules` 中。`module`的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
# tensor([[ 0.0251,  0.0574,  0.1081,  0.0849, -0.0683,  0.1211,  0.1951, -0.1257,
#           0.1838, -0.0198],
#         [ 0.3490,  0.0244,  0.1651,  0.2071,  0.0297,  0.0638,  0.0480, -0.0929,
#           0.2836, -0.0195]], grad_fn=<AddmmBackward0>)
```

在正向传播函数中执行代码

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
net(X)
# tensor(0.0105, grad_fn=<SumBackward0>)
```

混合搭配各种组合块的方法

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
# tensor(0.1289, grad_fn=<SumBackward0>)
```

## 参数管理

假设已经定义好自己的类，关注参数如何访问

首先关注具有单层隐藏层的多感知机层

```python
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
# tensor([[-0.1206],
#         [-0.0657]], grad_fn=<AddmmBackward0>)
```

参数访问，拿出每层的权重

```python
print(net[2].state_dict())
# OrderedDict([('weight', tensor([[-0.1599,  0.1461, -0.3413, -0.0212, -0.0406,  0.3361,  0.0051, -0.1932]])), ('bias', tensor([-0.1473]))])
```

目标参数，直接访问一个具体的参数

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# <class 'torch.nn.parameter.Parameter'>
# Parameter containing:
# tensor([-0.1473], requires_grad=True)
# tensor([-0.1473])
net[2].weight.grad == None # 这里等于None是因为还没有进行反向计算，所以没有梯度
# True
```

一次性访问所有参数

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
# ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
# ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
net.state_dict()['2.bias'].data
# tensor([-0.1473])
```

从嵌套块收集参数

```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
# tensor([[0.4448],
#         [0.4448]], grad_fn=<AddmmBackward0>)
print(rgnet)
# Sequential(
#   (0): Sequential(
#     (block 0): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 1): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 2): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#     (block 3): Sequential(
#       (0): Linear(in_features=4, out_features=8, bias=True)
#       (1): ReLU()
#       (2): Linear(in_features=8, out_features=4, bias=True)
#       (3): ReLU()
#     )
#   )
#   (1): Linear(in_features=4, out_features=1, bias=True)
# )
```

内置初始化

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
# (tensor([-0.0203,  0.0043,  0.0102, -0.0109]), tensor(0.))
```

还可以将所有的参数初始化为给定的常数，比如初始化为1

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
# (tensor([1., 1., 1., 1.]), tensor(0.))
```

对某些块应用不同的初始化方法

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
# tensor([-0.4006, -0.3161,  0.4433, -0.2799])
# tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
```

自定义初始化

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]

Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])

# Out[12]:
# tensor([[ 7.8307, -0.0000,  0.0000, -0.0000],
#         [ 8.7647,  6.9081, -7.3833,  8.2137]], grad_fn=<SliceBackward0>)
```

更简单的方法

```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
# tensor([42.,  1.,  1.,  1.])
```

参数绑定，比如想在几个层之间共享参数

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
# tensor([True, True, True, True, True, True, True, True])
# tensor([True, True, True, True, True, True, True, True])
```

## 自定义层

构造一个没有任何参数的自定义层

```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
# tensor([-2., -1.,  0.,  1.,  2.])
```

将层作为组件合并到构建更复杂的模型中

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
# tensor(-4.6566e-09, grad_fn=<MeanBackward0>)
```

带有参数的层

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
linear.weight
# Parameter containing:
# tensor([[-1.4141, -0.0254,  0.0690],
#         [ 1.2162,  0.4508,  1.6161],
#         [ 0.2743, -0.0254, -0.6869],
#         [ 0.3095, -0.1783, -0.3168],
#         [ 0.1938, -0.3268, -0.2852]], requires_grad=True)
```

使用自定义层直接执行正向传播计算

```python
linear(torch.rand(2, 5))
# tensor([[1.4513, 0.0000, 0.4147],
#         [1.4359, 0.0000, 0.3388]])
```

## 读写文件

加载和保存张量

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
x2

# tensor([0, 1, 2, 3])
```

存储一个张量列表，然后把它们读回内存

```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
# (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))
```

写入或读取从字符串映射到张量的字典

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
# {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}
```

加载和保存模型参数

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

将模型的参数存储为一个叫做mlp.params的文件

```python
torch.save(net.state_dict(), 'mlp.params')
```

实例化原始多层感知机模型的一个备份，直接读取文件中存储的参数

```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
# MLP(
#   (hidden): Linear(in_features=20, out_features=256, bias=True)
#   (output): Linear(in_features=256, out_features=10, bias=True)
# )
```

```python
Y_clone = clone(X)
Y_clone == Y
# tensor([[True, True, True, True, True, True, True, True, True, True],
#         [True, True, True, True, True, True, True, True, True, True]])
```

