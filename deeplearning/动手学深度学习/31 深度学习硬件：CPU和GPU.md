# 31 深度学习硬件：CPU和GPU

## 提升CPU利用率 1

在计算a+b之前，需要准备数据

- 主内存 > L3 >  L2 > L1 > 寄存器
  - L1访问延时：0.5 ns
  - L2访问延时：7 ns（14 * L1）
  - 主内存访问延时：100 ns（200 * L1）

提升空间和时间的内存本地性

- 时间：重复使用数据使得保持他们在缓存里
- 空间：按序读写数据使得可以预读取

## 样例分析

如果一个矩阵是按行存储，访问一行会比访问一列要快

- CPU一次读取64字节（缓存线）
- CPU会“聪明得”提前读取下一个缓存线

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-13 17.11.05.png" alt="截屏2021-12-13 17.11.05" style="zoom:50%;" />

## 提升CPU利用率 2

高端CPU有几十个核

并行来利用所有核

- 超线程不一定提升性能，因为它们共享寄存器

## 样例分析

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-13 17.14.24.png" alt="截屏2021-12-13 17.14.24" style="zoom:50%;" />

## CPU vs. GPU

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-13 17.20.26.png" alt="截屏2021-12-13 17.20.26" style="zoom:50%;" />

## 提升GPU利用率

并行

内存本地性

少用控制语句