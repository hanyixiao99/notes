### Why we need Explainable ML?

Correct answers != Intelligent

Loan issuers are required by law to explain their models.

Medical diagnosis model is responsible for human life. Can it be a black box?

If a model is used at the court, we must make sure the model behaves in a nondiscriminatory manner.

If a self-driving car suddenly acts abnormally, we need to explain why.

We can improve ML model based on explanation.

### Interpretable v.s. Powerful

Some models are intrinsically interpretable

- For example, linear model (from weights, you know the importance of features)
- But not very powerful.

Deep network is difficult to interpretable. Deep networks are black boxes ... but powerful than a linear model.



Are there some models interpretable and powerful at the same time? How about decision tree?

### Goal of Explainable ML

Make people comfortable.

## Explainable ML

### Local Explanation - Explain the Decision

Why do you think this image is a cat?

Which component is critical?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 16.39.08.png" alt="截屏2022-05-01 16.39.08" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 16.41.57.png" alt="截屏2022-05-01 16.41.57" style="zoom:33%;" />

another way

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 16.44.03.png" alt="截屏2022-05-01 16.44.03" style="zoom:33%;" />

Limitation: Noisy Gradient

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 16.51.08.png" alt="截屏2022-05-01 16.51.08" style="zoom:33%;" />

SmoothGrad: Randomly add noises to the input image, get saliency maps of the noisy images, and averange them. 1706.03825

Limitation: Gradient Saturation

Gradient cannot always reflect importance

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 16.54.36.png" alt="截屏2022-05-01 16.54.36" style="zoom:33%;" />

### How a network processes the input data?

#### Visualization

Colors: speakers

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 16.59.20.png" alt="截屏2022-05-01 16.59.20" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 17.00.06.png" alt="截屏2022-05-01 17.00.06" style="zoom:33%;" />

#### Probing

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 17.05.18.png" alt="截屏2022-05-01 17.05.18" style="zoom:33%;" />



### Global Explanation - Explain the Whole Model

What does a "cat" look like?

#### What does a filter detect?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-05-01 17.12.53.png" alt="截屏2022-05-01 17.12.53" style="zoom:33%;" />

Lets create an  image including the patterns

