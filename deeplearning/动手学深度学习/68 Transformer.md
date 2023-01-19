# 68 Transformer

## 架构

基于编码器-解码器架构来处理序列对

跟使用注意力的seq2seq不同，Transformer是纯基于注意力（self- attention）

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.51.55.png" alt="截屏2021-12-17 16.51.55" style="zoom:50%;" />

## 多头注意力

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.54.11.png" alt="截屏2021-12-17 16.54.11" style="zoom:50%;" />

对同一key，value，query，希望抽取不同的信息

- 例如短距离关系和长距离关系

多头注意力使用h个独立的注意力池化

- 合并各个头（head）输出得到最终输出

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.54.42.png" alt="截屏2021-12-17 16.54.42" style="zoom:50%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.56.11.png" alt="截屏2021-12-17 16.56.11" style="zoom:50%;" />

头i的可学习参数

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.56.27.png" alt="截屏2021-12-17 16.56.27" style="zoom:50%;" />

头i的输出

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.56.41.png" alt="截屏2021-12-17 16.56.41" style="zoom:50%;" />

输出的可学习参数

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.56.56.png" alt="截屏2021-12-17 16.56.56" style="zoom:50%;" />

多头注意力输出

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.57.56.png" alt="截屏2021-12-17 16.57.56" style="zoom:50%;" />

## 有掩码的多头注意力

解码器对序列中一个元素输出时，不应该考虑该元素之后的元素

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.59.29.png" alt="截屏2021-12-17 16.59.29" style="zoom:50%;" />

可以通过掩码来实现

- 也就是计算xi输出时，假设当前序列长度为i

## 基于位置的前馈网络（全连接层）

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 17.02.25.png" alt="截屏2021-12-17 17.02.25" style="zoom:50%;" />

将输入形状由(b,n,d)变换成(bn,d)

作用两个全连接层

输出形状由(bn,d)变化回(b,n,d)

等价于两层核窗口为1的一层卷积

## 层归一化

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 17.02.04.png" alt="截屏2021-12-17 17.02.04" style="zoom:50%;" />

批量归一化对每个特征/通道里元素进行归一化

- 不适合序列长度会变的NLP应用

层归一化对每个样本里的元素进行归一化

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 17.01.49.png" alt="截屏2021-12-17 17.01.49" style="zoom:50%;" />

## 信息传递

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 17.03.50.png" alt="截屏2021-12-17 17.03.50" style="zoom:50%;" />

编码器中的输出y1,...,yn

将其作为解码中第i歌Transformer块中多头注意力的key和value

- 它的query来自目标序列

意味着编码器和解码器中块的个数和输出维度都是一样的

## 预测

预测第t+1个输出时

解码器中输入前t个预测值

- 在自注意力中，前t个预测值作为key和value，第t个预测值还作为query

