## Local minima & Saddle Point

Optimization Fails because...

### Critical point

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 14.30.15.png" alt="截屏2022-04-16 14.30.15" style="zoom:33%;" />

#### Which one?

Tayler Series Approximation  

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 14.38.44.png" alt="截屏2022-04-16 14.38.44" style="zoom:33%;" />

AT Critical Point, g = 0, We got Hessian.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 14.44.42.png" alt="截屏2022-04-16 14.44.42" style="zoom:33%;" />

If in a saddle point , H may tell us parameter update direction.

#### Saddle Point v.s. Local Minima

Saddle Point most time.

## Batch & Monentum

### Batch

1 epoch = see all the batches once. Shuffle after each epoch.

#### Why batch?

When batch size = N (Full batch), model will update after seeing all of the examples. Long time for cooldown, but powerful.

When batch size = 1, update for each example. Short time for cooldown, but noisy.

#### Small Batch v.s. Large Batch

Smaller batch requires longer time for one epoch (longer time for seeing all data once).

But Smaller batch size has better performance. (Noisy update is better for training).

### Momentum

movement of last step minus gradient at present.

## Adaptive Learning Rate

People believe training stuck because the parameters are around a critical point ...

Learning rage cannot be one-size-fits-all.

#### Different parameters needs different learning rate

by RMSProp. The recent gradient has larger influence, and the past gardients have less influence.

#### Adam

RMSProp + Momentum

#### Learning Rate Scheduling

==Learning rate decay==. As the training goes, we are closer to the destination, so we reduce the learning rate.

==Warm up==. Increase and then decrease? (BERT)

## Loss

### Loss of Classification

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 16.26.11.png" alt="截屏2022-04-16 16.26.11" style="zoom:33%;" />

#### Mean Square Error (MSE)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 16.27.15.png" alt="截屏2022-04-16 16.27.15" style="zoom:33%;" />

#### Cross-entropy

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 16.27.23.png" alt="截屏2022-04-16 16.27.23" style="zoom:33%;" />

Minimizing cross-entropy is equivalent to maximizing likelihood.

With Softmax in PyTorch.

## Case Study: Pokémon v.s. Digimon

pass

 ## HW2

https://colab.research.google.com/drive/1hmTFJ8hdcnqRz_0oJSXjTGhZLVU-bS1a?usp=sharing

### Task Introduction

Data Preprocessing: Extract MFCC features from raw waveform (already done).

Classification: Perform framewise phoneme classification using pre-extracted MFCC features.

#### Task: Multiclass Classification

Framewise phoneme prediction from speech.

#### Data Preprocessing

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 17.32.20.png" alt="截屏2022-04-16 17.32.20" style="zoom:33%;" />

#### More Information About the Data

Since each frame only contains 25 ms of speech, a single frame is unlikely to represent a complete phoneme.

- Usually, a phoneme will span several frames

- Concatenate the neighboring phonemes for training

Finding testing labels or doing human labeling are strictly prohibited.

### Dataset & Data Format

#### Dataset

==LibriSpeech==

#### Data Format

Each .pt file is extracted from one original wav file

Use `torch.load()` to read in .pt files as torch tensors

Each tensor has a shape of (T, 39)
