# 54 循环神经网络RNN

<!--循环神经网络的输出取决于当下输入和前一时间的隐变量-->

<!--应用到语言模型中时，循环神经网络根据当前词预测下一次时刻词-->

<!--通常使用困惑度来衡量语言模型的好坏-->

潜变量自回归模型

- 使用潜变量ht总结过去信息

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.36.16.png" alt="截屏2021-12-16 16.36.16" style="zoom:50%;" />

## 循环神经网络

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.36.55.png" alt="截屏2021-12-16 16.36.55" style="zoom:50%;" />

更新隐藏状态：

这里fai是激活函数

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.37.25.png" alt="截屏2021-12-16 16.37.25" style="zoom:50%;" />

输出（图中有误，不需要激活函数，输出权重为Woh）：

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.37.41.png" alt="截屏2021-12-16 16.37.41" style="zoom:50%;" />

## 使用循环神经网络的语言模型

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.39.57.png" alt="截屏2021-12-16 16.39.57" style="zoom:50%;" />

输出发生在观察之前

## 困惑度（perplexity）

衡量一个语言模型的好坏可以用平均交叉熵

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.50.07.png" alt="截屏2021-12-16 16.50.07" style="zoom:50%;" />

其中p是语言模型的预测概率，xt是真实词

历史原因NLP使用困惑度exp(\pai)来衡量，是平均每次可能选项

- 1表示完美，无穷大是最差情况

## 梯度裁剪

迭代中计算这T个时间步上的梯度，在反向传播过程中产生长度为O(T)的矩阵乘法链，导致数值不稳定

梯度裁剪能有效预防梯度爆炸

- 如果梯度长度超过$\theta$，那么拖影回长度$\theta$

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.55.34.png" alt="截屏2021-12-16 16.55.34" style="zoom:50%;" />

## 更多的应用 RNNs

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-16 16.58.02.png" alt="截屏2021-12-16 16.58.02" style="zoom:50%;" />
