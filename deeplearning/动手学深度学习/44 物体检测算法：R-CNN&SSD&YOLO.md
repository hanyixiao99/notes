# 44 物体检测算法：R-CNN&SSD&YOLO

 ## 区域卷积神经网络R-CNN

<!--R-CNN是最早也是最有名的一类基于锚框和CNN的目标检测算法-->

<!--Fast/Faster R-CNN持续提升性能-->

<!--Faster R-CNN和Mask R-CNN是在求最高精度场景下的常用算法-->

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 15.54.48.png" alt="截屏2021-12-15 15.54.48" style="zoom:50%;" />

首先使用启发式（选择性）搜索算法来选择锚框

使用预训练模型来对每个锚框抽取特征

训练一个SVM来对类别分类

训练一个线性回归模型来预测边缘框偏移

### 兴趣区域（RoI）池化层（Fast-RCNN）

由于每次选定锚框的大小都不一样

给定一个锚框，均匀分割成n * m块，输出每块里的最大值

不管锚框多大，总是输出nm个值，变成同样大小进行batch小批量处理

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 15.58.14.png" alt="截屏2021-12-15 15.58.14" style="zoom:50%;" />

### Fast R-CNN

使用CNN对图片抽取特征（不是对每个锚框抽取特征，而是对整个图片抽取特征）

使用RoI池化层对每个锚框生成固定长度特征

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 16.07.09.png" alt="截屏2021-12-15 16.07.09" style="zoom:50%;" />

### Faster R-CNN

使用一个区域提议网络来替代启发式（选择性）搜索获得更好的锚框

- 给定低质量（结果较差）的锚框
- 通过二分类和NMS输出结果较好的锚框给后面的大网络用

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 16.09.03.png" alt="截屏2021-12-15 16.09.03" style="zoom:50%;" />

### Mask R-CNN

如果有像素级别的标号，使用FCN来利用这些信息

- 对像素做预测

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 16.15.23.png" alt="截屏2021-12-15 16.15.23" style="zoom:50%;" />

## 单发多框检测SSD

<!--SSD通过单神经网络来检测模型-->

<!--以每个像素为中心产生多个锚框-->

<!--在多个段的输出上进行多尺度的检测-->

### 生成锚框

对每个像素，生成多个以它为中心的锚框

给定n个大小s1-sn和m个高宽比，生成n+m-1个锚框，其大小和高宽比分别为

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 16.34.41.png" alt="截屏2021-12-15 16.34.41" style="zoom:50%;" />

### SSD模型

一个基础网络来抽取特征，然后多个卷积层块来减半高宽

在每段都生成锚框

- 底部段来拟合小物体，顶部段来拟合大物体

对每个锚框预测类别和边缘框

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 16.36.53.png" alt="截屏2021-12-15 16.36.53" style="zoom:50%;" />

## YOLO（You Only Look Once）

SSD中锚框大量重叠，因此浪费了很多计算

YOLO将图片均分成**S * S**个锚框，每个锚框预测**B**个边缘框（样本数S^2 * B，快)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-15 16.54.22.png" alt="截屏2021-12-15 16.54.22" style="zoom:50%;" />

假设边缘框有一定规律，可以使用聚类算法找出规律