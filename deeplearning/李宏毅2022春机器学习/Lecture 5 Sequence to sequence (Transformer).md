## Quick Introduction of Batch Normalization

https://www.bilibili.com/video/BV1Wv411h7kN?p=48

## Seq2seq

Transformer is a sequence-to-sequence (seq2seq) model

### Sequence-to-sequence (Seq2Seq)

Input a sequence, output a sequence.

The output length is determined by model.

Speech Recognition

Machine Translation

Speech Translation

Most NLP applications -> ==QA==

QA can be done by seq2seq

==question, context -> Seq2seq -> answer== 

1806.08730 1909.03329

#### Seq2seq for Syntactic Parsing

#### Seq2seq for Multi-label Classification

c.f. Mutil-class Classification

An object can belong to multiple classes.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 16.40.41.png" alt="截屏2022-04-18 16.40.41" style="zoom:33%;" />

1909.03434 1707.05495

#### Seq2seq for Object Detection

2005.12872

### Seq2seq

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 16.43.13.png" alt="截屏2022-04-18 16.43.13" style="zoom:33%;" />

### Encoder

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 16.45.48.png" alt="截屏2022-04-18 16.45.48" style="zoom:33%;" />

Normally 

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 16.48.47.png" alt="截屏2022-04-18 16.48.47" style="zoom:33%;" />

In Transformer 

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.04.32.png" alt="截屏2022-04-18 17.04.32" style="zoom:33%;" />

To learn more

2002.04745

2003.07845

### Decoder

2 types

#### Autoregressive (Speech Recognition as Example)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.15.11.png" alt="截屏2022-04-18 17.15.11" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.15.38.png" alt="截屏2022-04-18 17.15.38" style="zoom:33%;" />

##### Self-attention -> Masked Self-attention

When a^1 -> b^1, a^1 only see a^1,

When a^2 -> b^2, a^2 only see a^1 & a^2.

Self-attention:

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.21.10.png" alt="截屏2022-04-18 17.21.10" style="zoom:33%;" />

Masked self-attention:

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.22.11.png" alt="截屏2022-04-18 17.22.11" style="zoom:33%;" />

##### Why masked？

Consider how does decoder work (one by one).

##### Correct output length?

Adding "Stop Token"

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.27.56.png" alt="截屏2022-04-18 17.27.56" style="zoom:33%;" />


 We hope that

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.28.53.png" alt="截屏2022-04-18 17.28.53" style="zoom:33%;" />

#### Another Decoder: Non-autoregressive (NAT)

##### AT v.s. NAT

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.31.37.png" alt="截屏2022-04-18 17.31.37" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 17.32.15.png" alt="截屏2022-04-18 17.32.15" style="zoom:33%;" />

How to decide the output length for NAT decoder?

- Another predictor for output length
- Output a very long sequence, ignore tokens after END 

Advantage: parallel, controllable output length

NAT is usually worse than AT

### Encoder-Decoder

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 19.26.44.png" alt="截屏2022-04-18 19.26.44" style="zoom:33%;" />

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 19.29.34.png" alt="截屏2022-04-18 19.29.34" style="zoom:33%;" />

### Training

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 19.36.25.png" alt="截屏2022-04-18 19.36.25" style="zoom:33%;" />

==Teacher Forcing==: using the ground truth as input.

### Tips

#### Copy Mechanism

Chat-bot

Summarization

1704.04368

#### Guided Attention (TTS as example)

In some tasks, input and output are monotonically aligned.

Monotonic Attention, Location-aware Attention.

For example, speech recognition, TTS, etc.

#### Beam Search

Assume there are only two tokens (v=2).

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 19.55.37.png" alt="截屏2022-04-18 19.55.37" style="zoom:33%;" />

The red path is Greedy Decoding.

The green path is the best one. 

Not possible to check all the paths ... -> Beam Search

#### Optimizing Evaluation Metrics?

==BLEU socre==

When you don't know how to optimize, just use reinforcement learning (RL) 1511.06732

#### Exposure bias - Schduled Sampling

- Original Scheduled Sampling 1506.03099
- Scheduled Sampling for Transformer 1905.07651
- Parallel Scheduled Sampling 1906.04331

## Kind of Transformers

### How to make self-attention efficient?

Sequence length = N 

Attention Matrix = N * N (key * query)

### Notice

Self-attention is only a module in a larger network

Self-attention dominates computation when N is large

Usually developed for image processing (N = (256 * 256))

### Skip some calculations with human knowledge

Can we fill in some values with human knowledge?

### Local Attention / Truncated Attention

Similar with CNN

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 20.57.58.png" alt="截屏2022-04-18 20.57.58" style="zoom:33%;" />

### Stride Attention

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.00.34.png" alt="截屏2022-04-18 21.00.34" style="zoom:33%;" />

### Global Attention

Add special token into original sequence

- Attend to every token -> collect global information
- Attended by every token -> it knows global information

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.05.57.png" alt="截屏2022-04-18 21.05.57" style="zoom:33%;" />

### Many Different Choices ...

DIfferent heads ues different patterns

#### Longformer 2004.05150

Local Attention + Stride Attention + Global Attention

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.08.38.png" alt="截屏2022-04-18 21.08.38" style="zoom:33%;" />

#### Big Bird 2007.14062

Longformer + Random Attention

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.10.11.png" alt="截屏2022-04-18 21.10.11" style="zoom:33%;" />

### Can we only focus on Critical Parts?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.14.58.png" alt="截屏2022-04-18 21.14.58" style="zoom:50%;" />

How to quickly estimate the portion with small attention weights?

#### Clustering

Reformer

Routing Transformer 2003.05997

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.18.15.png" alt="截屏2022-04-18 21.18.15" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.20.00.png" alt="截屏2022-04-18 21.20.00" style="zoom:33%;" />

### Learnable Patterns

Sinkhorn Sorting Network

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.23.14.png" alt="截屏2022-04-18 21.23.14" style="zoom:33%;" />

A grid should be skipped or not is decided by another learned module

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.25.52.png" alt="截屏2022-04-18 21.25.52" style="zoom:33%;" />

### Do we need full attention matrix? Linformer

2006.04768

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.34.29.png" alt="截屏2022-04-18 21.34.29" style="zoom:33%;" />

#### how?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.36.28.png" alt="截屏2022-04-18 21.36.28" style="zoom:33%;" />

Can we reduce the number of queries ? -> Change output sequence length.

#### How reduce number of keys?

Compressed Attention 1801.10198

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.40.00.png" alt="截屏2022-04-18 21.40.00" style="zoom: 33%;" />

Linformer 2006.04768

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.40.48.png" alt="截屏2022-04-18 21.40.48" style="zoom:33%;" />

Linear combination of N vectors

### Attention Mechanism is three-matrix Multiplication

Lin

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.43.58.png" alt="截屏2022-04-18 21.43.58" style="zoom:33%;" />

if we ignore softmax

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-18 21.48.56.png" alt="截屏2022-04-18 21.48.56" style="zoom:33%;" />

The result is the same, but the calculation time is different.

put softmax back

## HW5

https://colab.research.google.com/drive/1Tlyk2vCBQ8ZCuDQcCSEWTLzr1_xYF9CL#scrollTo=Le4RFWXxjmm0

