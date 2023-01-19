# 66 使用注意力机制的seq2seq

## 动机

机器翻译中，每个生成的词可能相关于源句子中不同的词

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.19.37.png" alt="截屏2021-12-17 16.19.37" style="zoom:50%;" />

seq2seq模型中不能对此直接建模

## 加入注意力

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-17 16.21.58.png" alt="截屏2021-12-17 16.21.58" style="zoom:50%;" />

编码器对每次词的输出作为key和value（它们是同样的）

- 假设英语句子长为3，就会有3个key value pair
- key value pair 就是某个词的RNN输出

解码器RNN对上一个词的输出是query

注意力的输出和下一个词的词嵌入合并进入