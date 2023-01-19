# 56 门控循环单元 GRU

## 关注一个序列

不是每个观察值都是同等重要

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 18.53.51.png" alt="截屏2021-12-16 18.53.51" style="zoom:50%;" />

想只记住相关的观察需要：

- 能关注的机制（更新门）
- 能遗忘的机制（重置门）

## 门

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 18.59.42.png" alt="截屏2021-12-16 18.59.42" style="zoom:50%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 18.59.50.png" alt="截屏2021-12-16 18.59.50" style="zoom:50%;" />

## 候选隐状态

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 19.04.51.png" alt="截屏2021-12-16 19.04.51" style="zoom:50%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 19.05.08.png" alt="截屏2021-12-16 19.05.08" style="zoom:50%;" />

不是真正的隐状态

首先假设不去看Rt，就是之前RNN计算隐藏状态的方式，但是现在加了Rt按元素乘法

对样本来讲，Rt和Ht是长度一样的向量，所以可以按元素做乘法，并且由于Rt做了sigmoid，因此是一个0-1之间的值

如果Rt靠近0，那么括号内元素乘法后的结果就会变得像0，意味着忘掉上一个时刻的隐藏状态，如果全部为0则变为初始状态，即从这个时刻开始之前的信息全部遗忘，从0初始化开始，另一个极端情况则为全部为1，意味着所有情况信息全部保留

由于Rt是可以学习的，因此可以选择什么保留什么遗忘

## 隐状态

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 19.08.48.png" alt="截屏2021-12-16 19.08.48" style="zoom:50%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 19.08.59.png" alt="截屏2021-12-16 19.08.59" style="zoom:50%;" />

真正的隐状态

Zt也是0-1的数字，假设Zt都等于1，Ht=Ht-1，意味着不更新，将过去状态直接变成现在，忽略掉现在的x元素。Zt都等于0时，回到RNN情况，不再拿取过去状态，基本只考虑当前状态

选择多少过去多少现在