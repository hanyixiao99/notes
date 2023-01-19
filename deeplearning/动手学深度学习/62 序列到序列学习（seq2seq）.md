# 62 序列到序列学习（seq2seq）

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 20.41.53.png" alt="截屏2021-12-16 20.41.53" style="zoom:50%;" />

<!--Seq2seq从一个句子生成另一个句子-->

<!--编码器和解码器都是RNN-->

<!--将编码器最后时间隐状态来初始解码器隐状态来完成信息传递-->

<!--常用BLEU来衡量生成序列的好坏-->

## 机器翻译

给定一个源语言的句子，自动翻译成目标语言

这两个句子可以有不同的长度

## Seq2seq

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 20.44.32.png" alt="截屏2021-12-16 20.44.32" style="zoom:50%;" />

编码器是一个RNN，读取输入句子

- 可以双向（双向可以Encoder，但是不能Decoder）

解码器使用另外一个RNN来输出

## 编码器-解码器细节

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 20.48.42.png" alt="截屏2021-12-16 20.48.42" style="zoom:50%;" />

编码器是没有输出的RNN

编码器最后时间步的隐状态用作解码器的初始隐状态

## 训练

训练时解码器使用目标句子作为输入

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 20.51.08.png" alt="截屏2021-12-16 20.51.08" style="zoom:50%;" />

## 衡量生成序列的好坏BLEU

pn是预测中所有n-gram的精度

- 标签序列ABCDEF和预测序列ABBCD，有

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 20.51.59.png" alt="截屏2021-12-16 20.51.59" style="zoom:50%;" />

BLEU定义

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 20.52.17.png" alt="截屏2021-12-16 20.52.17" style="zoom:50%;" />