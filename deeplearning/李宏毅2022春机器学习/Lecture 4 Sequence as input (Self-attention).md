## Self-attention

Input is a set of vectors

### Vector Set as Input

### What is the output?

#### Each vector has a label (focus on it this time)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.24.30.png" alt="截屏2022-04-18 14.24.30" style="zoom:33%;" />

POS tagging

HW2

#### The Whole sequence has a label

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.24.39.png" alt="截屏2022-04-18 14.24.39" style="zoom:33%;" />



Sentiment analysis

Speaker (HW4)

#### Model decides the number of laberls itself

==seq2seq==

Translation (HW5)

### Sequence Labeling

Is it possible to consider the context?

FC can consider the neighbor

How to consider the whole sequence?

a window covers the whole sequence?

### Self-attention

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.31.54.png" alt="截屏2022-04-18 14.31.54" style="zoom:33%;" />

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.38.17.png" alt="截屏2022-04-18 14.38.17" style="zoom:33%;" />

Find the relevant vectors in a sequence

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.39.56.png" alt="截屏2022-04-18 14.39.56" style="zoom:33%;" />

How to get alpha?  

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.42.45.png" alt="截屏2022-04-18 14.42.45" style="zoom:33%;" />



<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.44.45.png" alt="截屏2022-04-18 14.44.45" style="zoom:33%;" />

also need

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.45.20.png" alt="截屏2022-04-18 14.45.20" style="zoom:33%;" />

then alpha -> softmax / ReLU -> alpha' (get ==attention score==).

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.48.52.png" alt="截屏2022-04-18 14.48.52" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.49.21.png" alt="截屏2022-04-18 14.49.21" style="zoom:33%;" />

In another view

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 14.57.27.png" alt="截屏2022-04-18 14.57.27" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.00.05.png" alt="截屏2022-04-18 15.00.05" style="zoom:33%;" />

then

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.01.55.png" alt="截屏2022-04-18 15.01.55" style="zoom:33%;" />

then

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.25.16.png" alt="截屏2022-04-18 15.25.16" style="zoom:33%;" />

  ### Multi-head Self-attention

DIfferent types of relevance (different q for different relevance)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.33.51.png" alt="截屏2022-04-18 15.33.51" style="zoom:33%;" />

### Positional Encoding

No position information in self-attention

Each position has a unique positional vector e^i

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.38.20.png" alt="截屏2022-04-18 15.38.20" style="zoom:33%;" />

 ### Self-attention for Speech

Truncated Self-attention

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.43.18.png" alt="截屏2022-04-18 15.43.18" style="zoom:33%;" />

### Self-attention for Image

An image can also be considered as a vector set.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.44.48.png" alt="截屏2022-04-18 15.44.48" style="zoom:33%;" />

Self-Attention GAN 1805.08318

DEtection Transformer (DETR) 2005.12872

### Self-attention v.s. CNN

CNN: Self-attention that can only attends in a receptive field

CNN is simplified self-attention

S-A: CNN with learnable receptive field

S-A is the complex version of CNN

1911.03584

### Self-attention V.s. RNN

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 15.51.25.png" alt="截屏2022-04-18 15.51.25" style="zoom:33%;" />

2006.16236

### Self-attention for Graph

Consider edge: only attention to connected nodes

## HW4

https://colab.research.google.com/drive/1gC2Gojv9ov9MUQ1a1WDpVBD6FOcLZsog?usp=sharing

 ### Task Introduction 

Self-attention

### Speaker Identification

Classification

### Dataset

VoxCeleb2

