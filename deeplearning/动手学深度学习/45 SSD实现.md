# 45 SSD实现

## 多尺度目标检测

```python
%matplotlib inline
import torch
from d2l import torch as d2l

img = d2l.plt.imread('./img/catdog.jpg')
h, w = img.shape[:2]
h, w
# (561, 728)
```

在特征图（`fmap`）上生成锚框（`anchors`），每个单位（像素）作为锚框的中心

```python
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # ratios:一个是1:1，一个是高为宽两倍，一个高为宽0.5倍
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

探测小目标

```python
# 假设特征宽高为4，s=[0.15]意思占原图片宽高比15%的区域
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 17.14.39.png" alt="截屏2021-12-15 17.14.39" style="zoom:50%;" />

将特征图的高度和宽度减小一半，然后使用较大的锚框来检测较大的目标

```python
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 17.24.44.png" alt="截屏2021-12-15 17.24.44" style="zoom:50%;" />

将特征图的高度和宽度减小一半，然后将锚框的尺度增加到0.8

```python
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 17.27.11.png" alt="截屏2021-12-15 17.27.11" style="zoom:50%;" />

## 单发多框检测（SSD）

类别预测层

```python
%matplotlib inline
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 输入通道数，锚框数，类别数
def cls_predictor(num_inputs, num_anchors, num_classes):
		# num_classes + 1，背景类
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

边界框预测层

```python
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

连接多尺度的预测

```python
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
# (torch.Size([2, 55, 20, 20]), torch.Size([2, 33, 10, 10]))
```

首先将4D变为2D

```python
def flatten_pred(pred):
  	# 将通道数放到最后，start_dim=1表示从第二个维度开始拉成向量
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
  	# 在宽上concat
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
 
concat_preds([Y1, Y2]).shape
# torch.Size([2, 25300])
```

网络的定义，网络可以任取，通常用pre-train好的模型作为目标检测算法中CNN的模型

例子，高和宽减半块

```python
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

```python
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
# torch.Size([2, 10, 10, 10])
```

基本网络块BaseNet，从原始图片抽特征直到第一次做锚框的中间过程

```python
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
# torch.Size([2, 64, 32, 32])
```

完整的单发多框检测模型由5个模块组成，在5个尺度做目标检测

```python
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
      	# 因为这个例子的数据集很小就维持通道数不变
        blk = down_sample_blk(128, 128)
    return blk
```

为每个块定义前向计算

```python
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
  	# 在正常前向计算的基础上增加锚框的处理
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

超参数

```python
sizes = [[0.2, 0.272],  # 0.2*0.37 = 0.272^2
         [0.37, 0.447],  # 0.37*0.54 = 0.447^2
         [0.54, 0.619], 
         [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

定义完整的模型

```python
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i=get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self,'blk_%d'%i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

创建一个实例模型，然后用它执行前向计算

```python
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
# output anchors: torch.Size([1, 5444, 4])
# output class preds: torch.Size([32, 5444, 2])
# output bbox preds: torch.Size([32, 21776])
```

读取香蕉检测数据集

```python
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

初始化其参数并定义优化算法

```python
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

定义损失函数和评价函数

```python
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

训练模型

```python
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # 训练精确度的和，训练精确度的和中的示例数
    # 绝对误差的和，绝对误差的和中的示例数
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # 生成多尺度的锚框，为每个锚框预测类别和偏移量
        anchors, cls_preds, bbox_preds = net(X)
        # 为每个锚框标注类别和偏移量
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # 根据类别和偏移量的预测和标注值计算损失函数
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

预测目标

```python
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

得到预测边界框

```python
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

筛选置信度不低于0.9的边界框

```python
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```