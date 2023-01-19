# 09 Softmax回归&损失函数&图片分类数据集

## 回归 vs. 分类

回归估计一个连续值

分类预测一个离散类别

- MNIST：手写数字识别（10类）

- ImageNet：自然物体分类（1000类）

## Kaggle上的分类问题

- 将人类蛋白质显微镜图片分成28类

- 将恶意软件分成9个类别

- 将恶意的Wikipedia评论分成7类

## 从回归到多类分类

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-06 16.15.23.png" alt="截屏2021-12-06 16.15.23" style="zoom: 33%;" /><img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-06 16.17.00.png" alt="截屏2021-12-06 16.17.00" style="zoom:33%;" />

### 均方损失

对类别进行一位有效编码，使用均方损失训练，最大化置信度值作为预测
$$
\hat{y}=arg\max_io_i
$$

### 无校验比例

需要更置信地识别正确类（大余量）
$$
o_y-o_i\geq \Delta(y,i)
$$

### 校验比例

输出匹配概率（非负，和为1）
$$
\hat{\pmb{y}}=\mathrm{\mathrm{softmax}}(\pmb{o}) \\ 
\hat{y_i}=\frac{\mathrm{\mathrm{exp}}(o_i)}{\sum_k\mathrm{\mathrm{exp}}(o_k)}
$$
概率 $\pmb{y}$ 和 $\hat{\pmb{y}}$ 的区别作为损失

## Softmax和交叉熵损失

交叉熵常用来衡量两个概率的区别 $H(\pmb{p},\pmb{q})=\sum_i-p_i\mathrm{\mathrm{log}}(q_i)$

将它作为损失
$$
L(\pmb{y},\hat{\pmb{y}})=-\sum_iy_i\mathrm{\mathrm{log}}(\hat{y_i})=-\mathrm{\mathrm{log}}\hat{y_y}
$$
其梯度是真实概率和预测概率的区别
$$
\partial_{o_i}L(\pmb{y},\hat{\pmb{y}})=\mathrm{\mathrm{softmax}}(\pmb{o})_i-y_i
$$

## Softmax总结

Softmax回归是一个多类分类模型

使用Softmax操作子得到每个类的预测置信度

使用交叉熵来衡量预测和标号的区别作为损失函数

## 损失函数

<font color=\#8A2BE2>——</font>：损失函数 $y=0$ 时，变换预测值 ${y}'$ 

<font color=\#7CFC00> ——</font>：似然函数，$e^{-L}$

<font color=#FFB61E>——</font>：损失函数的梯度

### L2 Loss

$$
L(y,{y}')=\frac{1}{2}(y-{y}')^2
$$

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-06 16.49.23.png" alt="截屏2021-12-06 16.49.23" style="zoom:50%;" />

### L1 Loss

$$
L(y,{y}')=|y-{y}'|
$$

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-06 16.57.01.png" alt="截屏2021-12-06 16.57.01" style="zoom:50%;" />

 

### Huber's Robust Loss

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-06 17.24.28.png" alt="截屏2021-12-06 17.24.28" style="zoom:50%;" />

## 图片分类数据集

MNIST数据集是图像分类中广泛使用的数据集之一，但作为基准数据集过于简单，所以使用类似但更复杂的Fashion-MNIST数据集

```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt

d2l.use_svg_display()
```

通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中

```python
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
  root='./data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
  root='./data', train=False, transform=trans, download=True)

len(mnist_train), len(mnist_test)
# 60000, 10000
```

```python
mnist_train[0][0].shape
# torch.Size([1, 28, 28])
```

两个可视化数据集的函数

```python
def get_fashion_mnist_labels(labels):
  	"""返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 
                   'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(
  imgs, num_rows, num_cols, titles=None, scale=1.5):
  	"""Plot a list of images"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(
      num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
          	# 张量图片
            ax.imshow(img.numpy())
        else:
          	# PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

几个样本的图像及其相应的标签

```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 
  2, 9, titles=get_fashion_mnist_labels(y))
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/image-20211206175641811.png" alt="image-20211206175641811" style="zoom:50%;" />

读取一小批量数据，大小为`batch_size`

```python
batch_size = 256

def get_dataloader_workers():
  	"""使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
# 1.94 sec
```

定义`load_data_fashion_mnist`函数

```python
def load_data_fashion_mnist(batch_size, resize=None):  
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
      root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
      root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

## Softmax回归的从零开始实现

Softmax的细节

```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

将展平每个图像，将他们视为长度为784的向量

因为我们的数据集有10个类别，所以网络输出维度为10

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

给定一个矩阵`X`，我们可以对所有元素求和

```python
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))
# tensor([[5., 7., 9.]]) tensor([[ 6.],
#         [15.]])
```

实现Softmax
$$
\mathrm{\mathrm{Softmax}}(\pmb{X})_{ij}=\frac{\mathrm{\mathrm{exp}}(\pmb{X}_{ij})}{\sum_k\mathrm{\mathrm{exp}}(\pmb{X}_{ik})}
$$

```python
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 广播机制
```

我们将每个元素变成一个非负数，此外根据概率原理，每行总和为1

```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(1))
# tensor([[0.2051, 0.0229, 0.2100, 0.0406, 0.5215],
#         [0.2571, 0.1206, 0.0429, 0.1702, 0.4092]])
# tensor([1.0000, 1.0000])
```

实现Softmax回归模型

```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

创建一个数据`y_hat`，其中包含2个样本在3个类别的预测概率，使用`y`作为`y_hat`中的概率索引

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])
# tensor([0.1000, 0.5000])
```

实现交叉熵损失函数

```python
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

 print(cross_entropy(y_hat, y))
# tensor([2.3026, 0.6931])
```

将预测类别与真实`y`元素进行比较

```python
def accuracy(y_hat, y):
  	"""计算预计正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

print(accuracy(y_hat, y) / len(y))
# 0.5
```

我们可以评估在任意模型`net`的准确率

```python
def evaluate_accuracy(net, data_iter):
  	"""计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval() # 将模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

`Accumulator`实例中创建了两个变量，用于分别存储正确预测的数量和预测的总数量

```python
class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

```python
print(evaluate_accuracy(net, test_iter))
# 报错加上 if __name__ == '__main__':
# 0.0478
```

Softmax回归的训练

```python
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
```

定义一个在动画中绘制数据的实用程序类

```python
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        plt.draw()
        plt.pause(0.001)
        display.clear_output(wait=True)
```

训练函数

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], 
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
```

小批量随机梯度下降来优化模型的损失函数

```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

训练模型10个迭代周期

```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# PyCharm不显示迭代过程的动图，把Tools里SciView的勾取消
```

对图像进行分类预测

```python
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

![image-20211206212809845](/Users/hanyixiao/Library/Application Support/typora-user-images/image-20211206212809845.png)

## Softmax回归的简洁实现

通过深度学习框架的高级API能够使实现Softmax回归变得更加容易

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

Softmax回归的输出层是一个全连接层

```python
# PyTorch不会隐式地调整输入的形状
# 因此定义了展平层（flatten）在线性层前调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

在交叉熵损失函数中传递未归一化的预测，并同时计算Softmax及其对数

```python
loss = nn.CrossEntropyLoss()
```

使用学习率为0.1的小批量随机梯度下降作为优化算法

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

调用之前定义的训练函数来训练模型

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```
