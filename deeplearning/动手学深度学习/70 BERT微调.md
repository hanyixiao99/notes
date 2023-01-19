# 70 BERT微调

BERT对每一个词元返回抽取了上下文信息的特征向量

不同的任务使用不同的特征

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-20 19.12.38.png" alt="截屏2021-12-20 19.12.38" style="zoom:50%;" />

## 句子分类

将<cls>对应的向量输入到全连接层分类

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-20 19.14.19.png" alt="截屏2021-12-20 19.14.19" style="zoom:50%;" />

## 命名实体识别

识别一个词元是不是命名实体，例如人名、机构、位置等

将非特殊词元放进全连接层分类

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-20 19.16.24.png" alt="截屏2021-12-20 19.16.24" style="zoom:50%;" />

## 问题回答

给定一个问题和描述文字，从描述文字中找出一个片段作为回答

对片段中的每个词元预测它是不是回答的开头或结束

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-20 19.17.44.png" alt="截屏2021-12-20 19.17.44" style="zoom:50%;" />

 