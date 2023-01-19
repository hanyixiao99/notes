# 27 含并行连结的网络GoogLeNet / Inception V3

<!--Inception块用4条有不同超参数的卷积层和池化层的路来抽取不同的信息-->

<!--它的一个主要优点是模型参数小，计算复杂度低-->

<!--GoogLeNet使用了9个Inception块，是第一个达到上百层的网络-->

## 最好的卷积层超参数？

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 22.34.48.png" alt="截屏2021-12-12 22.34.48" style="zoom:50%;" />

 ## Inception块

4个路径从不同层面抽取信息，然后在输出通道维合并

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 22.37.25.png" alt="截屏2021-12-12 22.37.25" style="zoom:50%;" /><img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 22.41.33.png" alt="截屏2021-12-12 22.41.33" style="zoom:50%;" />

与单3x3或5x5卷积层比，Inception块有更少的参数个数和计算复杂度

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 22.45.04.png" alt="截屏2021-12-12 22.45.04" style="zoom:50%;" />

## GoogLeNet

5段，9个Inception块，大量使用1x1卷积层作为全连接层

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 22.58.27.png" alt="截屏2021-12-12 22.58.27" style="zoom:50%;" />

### Stage 1 & 2

更小的宽口，更多的通道

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 23.00.33.png" alt="截屏2021-12-12 23.00.33" style="zoom:50%;" />

### Stage 3

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 23.03.10.png" alt="截屏2021-12-12 23.03.10" style="zoom:50%;" />

### Stage 4 & 5

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 23.04.48.png" alt="截屏2021-12-12 23.04.48" style="zoom:50%;" />

## Inception有各种后续变种

Inception-BN（V2） 使用 batch normalization

Inception-V3 修改了Inception块

- 替换5x5为多个3x3卷积层
- 替换5x5为1x7和7x1卷积层
- 替换3x3为1x3和3x1卷积层
- 更深

Inception-V4 使用残差连接

## Inception V3  Stage 3

![截屏2021-12-12 23.16.44](/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 23.16.44.png)

## Inception V3 Stage 4

![截屏2021-12-12 23.17.11](/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 23.17.11.png)

## Inception V3 Stage 5

![截屏2021-12-12 23.17.36](/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-12 23.17.36.png)