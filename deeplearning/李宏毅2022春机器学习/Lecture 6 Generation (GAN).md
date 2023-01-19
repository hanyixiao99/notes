## Network as Generator

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 14.57.37.png" alt="截屏2022-04-19 14.57.37" style="zoom:33%;" />

Distribution: We know its formulation, so we can sample from it.

y -> complex distribution

### Why distribution?

The same input has different outputs.

Especially for the tasks needs "creativity"

- Drawing 
- Chatbot

## Generative Adversarial Network (GAN)

### Anime Face Generation

#### Unconditional generation

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.12.51.png" alt="截屏2022-04-19 15.12.51" style="zoom:33%;" />

### Discriminator

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.33.04.png" alt="截屏2022-04-19 15.33.04" style="zoom:33%;" />

Discriminator is a nerual network (that is, a function).

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.33.44.png" alt="截屏2022-04-19 15.33.44" style="zoom:33%;" />

### Basic Idea of GAN

Dead leaf butterfly.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.36.25.png" alt="截屏2022-04-19 15.36.25" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.38.56.png" alt="截屏2022-04-19 15.38.56" style="zoom:33%;" />

This is where the term "adversarial" comes from.

 ### Algorithm

Initialize generator and discriminator ==G & D==

In each training iteration:

**Step 1**: Fix generator G, and update discriminator D

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.52.06.png" alt="截屏2022-04-19 15.52.06" style="zoom:33%;" />

Discriminator learns to assign high scores to real objects and low scores to generated objects.

**Step 2**: Fix discriminator D, and update generator G

Generator learns to "fool" the discriminator

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.56.07.png" alt="截屏2022-04-19 15.56.07" style="zoom:33%;" />

Use ==gradient ascent==.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 15.58.51.png" alt="截屏2022-04-19 15.58.51" style="zoom:33%;" />

## Theory behind GAN

### Our Objective 

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.14.04.png" alt="截屏2022-04-19 16.14.04" style="zoom:33%;" />



<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.23.27.png" alt="截屏2022-04-19 16.23.27" style="zoom:33%;" />

How to compute the divergence?

### Sampling is good enough ...

Although we do not know the distributions of PG and Pdata, we can sample from them.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.28.38.png" alt="截屏2022-04-19 16.28.38" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.28.59.png" alt="截屏2022-04-19 16.28.59" style="zoom:33%;" />

### Discriminator 

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.30.35.png" alt="截屏2022-04-19 16.30.35" style="zoom:33%;" />

The Value(max V (D, G)) is related to JS divergence.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.33.07.png" alt="截屏2022-04-19 16.33.07" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.37.36.png" alt="截屏2022-04-19 16.37.36" style="zoom:33%;" />

so

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.38.50.png" alt="截屏2022-04-19 16.38.50" style="zoom:33%;" />

Replace

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.39.17.png" alt="截屏2022-04-19 16.39.17" style="zoom:33%;" />

## Tips for GAN

### JS divergence is not suitable

In most cases, PG and Pdata are not overlapped.

1. The nature of data

   Both Pdata and PG are low-dim manifold in high-dim space.

   <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.48.35.png" alt="截屏2022-04-19 16.48.35" style="zoom:33%;" />

   

   The overlap can be ignored 

2. Sampling

   Even though Pdata and PG have overlap.

   If you do not have enough sampling...

   <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.50.35.png" alt="截屏2022-04-19 16.50.35" style="zoom:33%;" />

#### What is the problem of JS divergence?

JS divergence is always log2 if two distributions do not overlap.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.52.12.png" alt="截屏2022-04-19 16.52.12" style="zoom:33%;" />

Intuition: If two distributions do not overlap, binary classifier achieves 100% accuracy.

The accuracy (or loss) means nothing during GAN training.

### Wasserstein distance

Considering one distribution P as a pile of earth, and another distribution Q as the target.

The average distance the Earth Mover has to move the earth.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.58.13.png" alt="截屏2022-04-19 16.58.13" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 16.59.15.png" alt="截屏2022-04-19 16.59.15" style="zoom:33%;" />

Using the "moving plan" with the samllest average distance to define the Wasserstein distance. 

### WGAN

Evaluate Wasserstein distance between Pdata and PG

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 17.16.58.png" alt="截屏2022-04-19 17.16.58" style="zoom:33%;" />

D has to be smooth enough.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 17.18.53.png" alt="截屏2022-04-19 17.18.53" style="zoom:33%;" />

#### Spectral Normalization (SNGAN)

keep gradient norm smaller than 1 everywhere to smooth.

### GAN is still challenging ...

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 17.27.02.png" alt="截屏2022-04-19 17.27.02" style="zoom:33%;" />

### More Tips

github.com/soumith/ganhacks

1511.06434

1606.03498

1809.11096

## GAN for Sequence Generation

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 17.32.06.png" alt="截屏2022-04-19 17.32.06" style="zoom:33%;" />

 Usually, the generator are fine-tuned from a model learned by other approcahes.

However, with enough hyper parameter-tuning and tips, ==ScartchGAN== (1905.09922) can train from scratch.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 17.36.04.png" alt="截屏2022-04-19 17.36.04" style="zoom:33%;" />

  ## Evaluation of Generation

### Quality of Image

Human evaluation is expensive (and sometimes unfair/unstable).

How to evaluate the quality of the generated images automatically?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 17.53.21.png" alt="截屏2022-04-19 17.53.21" style="zoom:33%;" />

#### Diversity - Mode Collapse

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 18.51.42.png" alt="截屏2022-04-19 18.51.42" style="zoom:33%;" />

#### Diversity - Mode Dropping

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 18.54.48.png" alt="截屏2022-04-19 18.54.48" style="zoom:33%;" />

### Diversity

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 18.56.28.png" alt="截屏2022-04-19 18.56.28" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 18.56.50.png" alt="截屏2022-04-19 18.56.50" style="zoom:33%;" />

**Inception Score** (IS): Good quality, large diversity -> large IS

**Frechet Inception Distance** (FID)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.01.41.png" alt="截屏2022-04-19 19.01.41" style="zoom:33%;" />

Smaller is better

### We dont want memory GAN

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.04.28.png" alt="截屏2022-04-19 19.04.28" style="zoom:33%;" />

### To learn more about evaluation ...

1802.03446

## Conditional Generation

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.06.36.png" alt="截屏2022-04-19 19.06.36" style="zoom:33%;" />

### Conditional GAN

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.09.10.png" alt="截屏2022-04-19 19.09.10" style="zoom:33%;" />

So, improve

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.11.59.png" alt="截屏2022-04-19 19.11.59" style="zoom:33%;" />

more, Image Translation

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.13.26.png" alt="截屏2022-04-19 19.13.26" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.18.47.png" alt="截屏2022-04-19 19.18.47" style="zoom:33%;" />

## Learning from Unpaired Data

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.25.08.png" alt="截屏2022-04-19 19.25.08" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.26.17.png" alt="截屏2022-04-19 19.26.17" style="zoom:33%;" />

Can we learn the mapping without any paired data?

Unsupervised Conditional Generation

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.28.15.png" alt="截屏2022-04-19 19.28.15" style="zoom:33%;" />

 

### Cycle GAN

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 19.37.29.png" alt="截屏2022-04-19 19.37.29" style="zoom:33%;" />

same as Disco GAN, Dual GAN.

