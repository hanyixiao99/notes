<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-19 20.44.02.png" alt="截屏2022-04-19 20.44.02" style="zoom:33%;" />

  The models become larger and larger ...

## BERT

### Self-supervised Learning

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 14.42.19.png" alt="截屏2022-04-20 14.42.19" style="zoom:33%;" />

### Masking Input

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 14.54.10.png" alt="截屏2022-04-20 14.54.10" style="zoom:33%;" />

BERT is a Transformer Encoder

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 14.54.48.png" alt="截屏2022-04-20 14.54.48" style="zoom:33%;" />

Randomly masking some tokens

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 14.55.51.png" alt="截屏2022-04-20 14.55.51" style="zoom:33%;" />

Classification

### Next Sentence Perdiction

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 14.58.29.png" alt="截屏2022-04-20 14.58.29" style="zoom:33%;" />

<!--This approach is not helpful-->

<!--Robustly optimized BERT approach (RoBERTa) 1907.11692-->

**SOP**: Sentence order prediction

Used in ALBERT 1908.11942

### How to use BERT?

Pre-train

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 15.04.24.png" alt="截屏2022-04-20 15.04.24" style="zoom:33%;" />



<u>Downstream Tasks</u>

The tasks we care, we have a little bit labeled data.

 ### GLUE

General Language Understanding Evaluation (GLUE)

[gluebenchmark.com](https://gluebenchmark.com/)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.05.17.png" alt="截屏2022-04-20 16.05.17" style="zoom:33%;" />

Chinese version 

[cluebenchmarks.com](https://cluebenchmarks.com/)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.07.34.png" alt="截屏2022-04-20 16.07.34" style="zoom:33%;" />

### How to use BERT - Case 1

Input: sequence Output: class

Example: Sentiment analysis

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.11.40.png" alt="截屏2022-04-20 16.11.40" style="zoom:33%;" />

This is the model to be learned.

Why init by per-train? Better than random.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.13.22.png" alt="截屏2022-04-20 16.13.22" style="zoom:33%;" />

### How to use BERT - Case 2

Input: sequence Output: sequence (same as input)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.15.28.png" alt="截屏2022-04-20 16.15.28" style="zoom:33%;" />

Example: POS tagging

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.16.17.png" alt="截屏2022-04-20 16.16.17" style="zoom:33%;" />

### Case 3

Input: two sequences Output: a class

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.18.31.png" alt="截屏2022-04-20 16.18.31" style="zoom:33%;" />

Example: Natural Language Inferencee (NLI)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.19.31.png" alt="截屏2022-04-20 16.19.31" style="zoom:33%;" />

### Case 4

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.20.34.png" alt="截屏2022-04-20 16.20.34" style="zoom:33%;" />

Extraction-based Question Answering (QA)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.21.53.png" alt="截屏2022-04-20 16.21.53" style="zoom:33%;" />



<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.40.20.png" alt="截屏2022-04-20 16.40.20" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.40.46.png" alt="截屏2022-04-20 16.40.46" style="zoom:33%;" />

### Why does BERT work?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.52.08.png" alt="截屏2022-04-20 16.52.08" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.52.50.png" alt="截屏2022-04-20 16.52.50" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 16.53.46.png" alt="截屏2022-04-20 16.53.46" style="zoom:33%;" />

### Multi-lingual BERT

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.03.54.png" alt="截屏2022-04-20 17.03.54" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.04.32.png" alt="截屏2022-04-20 17.04.32" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.06.42.png" alt="截屏2022-04-20 17.06.42" style="zoom:33%;" />

## GPT

Predict Next Token

### How to use GPT?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.19.18.png" alt="截屏2022-04-20 17.19.18" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.20.16.png" alt="截屏2022-04-20 17.20.16" style="zoom:33%;" />

## Beyond Text

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.21.46.png" alt="截屏2022-04-20 17.21.46" style="zoom:33%;" />

## Recent Advances in Pre-trained Language Models

### Background knowledge

#### Pre-trained Language Models

==Neural Language Models==: A neural network that defines the probability over sequences of words

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.28.45.png" alt="截屏2022-04-20 17.28.45" style="zoom:33%;" />

##### How are these language models trained?

Given an incomplete sentence, predict the rest of the sentence

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.30.12.png" alt="截屏2022-04-20 17.30.12" style="zoom:33%;" />

##### Autoregerssive Language Models (ALMs): Complete the sentence given its prefix

Sentence completion

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.32.11.png" alt="截屏2022-04-20 17.32.11" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.32.42.png" alt="截屏2022-04-20 17.32.42" style="zoom:33%;" />

##### Transformer-based ALMs: Composed of stacked layers of transformer layers

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.35.10.png" alt="截屏2022-04-20 17.35.10" style="zoom:33%;" />

##### Training a language model is self-supervised learning

Self-supervised learning: Predicting any part of the input from any other part

##### Masked Language Models (MLMs): Use the unmasked words to predict the masked word

Cloze

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.38.54.png" alt="截屏2022-04-20 17.38.54" style="zoom:33%;" />

##### ==Pre-trained Language Models (PLMs)==

Using a large corpora to train a neural language model

- Autoregressive pre-trained: GPT
- MLM-based pre-trained: BERT

We believe that after per-training, the PLM learns some knowledge, encoded in its hidden repersentations, that can transfer to downstream tasks.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.42.31.png" alt="截屏2022-04-20 17.42.31" style="zoom:33%;" />

==(Standard) fine-tuning==: Using the per-trained weights of the PLM to initialize a model for a downstream task.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.45.07.png" alt="截屏2022-04-20 17.45.07" style="zoom:33%;" />

Fine-tuning PLMs on downstream tasks achieves exceptional performacne on many kinds of downstream tasks.

PLMs are widely applied to many different scenarios in different realms.

The next goal is to make PLMs fit in real-life use case

- How unrealistic is PLMs nowadays?

### The Problems of PLMs

#### Problem 1: Data scarcity in downstream tasks

A large amount of labeled data is not easy to obtain for each downstream task.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.50.38.png" alt="截屏2022-04-20 17.50.38" style="zoom:33%;" />

#### Problem 2: The PLM is too big, and they are still gitting bigger

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.51.21.png" alt="截屏2022-04-20 17.51.21" style="zoom:33%;" />

Need a copy for each downstream task

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.52.52.png" alt="截屏2022-04-20 17.52.52" style="zoom:33%;" />

Inference takes too long

Consume too much space

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.53.44.png" alt="截屏2022-04-20 17.53.44" style="zoom:33%;" />

### The Solutions of Those Problems

#### Labeled Data Scarcity -> Data-Efficient Fine-tuning

##### Prompt Tuning

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.57.35.png" alt="截屏2022-04-20 17.57.35" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.58.12.png" alt="截屏2022-04-20 17.58.12" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 17.58.51.png" alt="截屏2022-04-20 17.58.51" style="zoom:33%;" />

By converting the data points in the dataset into natural language prompts, the model may be easier to know what it should do.

Format the downstream task as a language modelling task with pre-defined templates into natural language prompts.

What you need in prompt tuning?

- A prompt template

  Convert data points into a natural language prompt

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.02.57.png" alt="截屏2022-04-20 18.02.57" style="zoom:33%;" />

- A PLM

  Perform language modeling

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.04.11.png" alt="截屏2022-04-20 18.04.11" style="zoom:33%;" />

- A verbalizer

  A mapping between the label and the vocabulary

  - Which vocabulary should repersents the class "entailment"

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.05.36.png" alt="截屏2022-04-20 18.05.36" style="zoom:33%;" />

The whole PLM will be fine-tuned

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.07.20.png" alt="截屏2022-04-20 18.07.20" style="zoom:33%;" />

Prompt tuning has better performance under data scarcity decause

- It incorporates human knowledge
- It introduces no new parameters

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.10.16.png" alt="截屏2022-04-20 18.10.16" style="zoom:33%;" />

Lets see how prompts can help us under different level of data scarcity.

##### Few-shot Learning

We have some(10) labeled training data.

GPT-3 can be used for few-shot setting, but GPT-3 is not freely available and contains 175B parameters.

Can we use smaller(?) PLMs and make them to perform well in few-shot learning?

- ==LM-BFF==: **b**etter **f**ew-shot **f**ine-tuning of **l**anguage **m**odels

  ​               Alternatively, language models' **b**est **f**riends **f**orever.

  - Core concept: prompt + demonstration

    <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.18.27.png" alt="截屏2022-04-20 18.18.27" style="zoom:33%;" />

  - Prompt tuning: No new parameters are introduced during fine-tuning

  - Automatic template searching

    <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.21.28.png" alt="截屏2022-04-20 18.21.28" style="zoom:33%;" />

##### Semi-supervised Learning

We have some labeled training data and a large amount of unlabeled data

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.27.47.png" alt="截屏2022-04-20 18.27.47" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.28.44.png" alt="截屏2022-04-20 18.28.44" style="zoom:33%;" />

 

==Pattern-Exploiting Training (PET)==

- Step 1: Use different prompts and verbalizer to prompt-tune different PLMs on the label dataset

    <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.32.20.png" alt="截屏2022-04-20 18.32.20" style="zoom:33%;" />

- Step 2: Predict the unlabeled dataset and combine the predictions from different models

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.33.27.png" alt="截屏2022-04-20 18.33.27" style="zoom:33%;" />

- Step 3: Use a PLM with classifier head to train on the soft-labeled dataset

  <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-20 18.34.24.png" alt="截屏2022-04-20 18.34.24" style="zoom:33%;" />

##### Zero-shot Learning

Zero-shot inference: inference on the downsteam task without any training data

If you don't have training data, then we need a model that can zero-shor inference on downstream tasks.

GPT-3 shows that zero-shot (with task description) is possible, only if your model is large enough.

Where does this zero-shot ability spring from?

- Hypothesis: during per-training, the training datasets implicitly contains a mixture of different tasks.
  - QA
  - Summarization

- Hypothesis: Multi-task training enables zero-shot generalization

  Why not train a model with multi-task learning on a bunch of dataset?

Multi-task fine-tuning using a PLM

- Convert the task into a natural language prompts
- Example: Natural Language Inference

Fine-tunging with some types of tasks and zero-shot inference on other types of tasks

Sometimes achieves performance better than GPT-3(175B parameters) with only 11B parameters

##### Summary

Use natural language prompts and add scenario-specific designs

#### PLMs Are Gigantic -> Reducing the Number of Parameters

Problem: PLM is too large (in terms of numbers of parameters, model size, and the storage needed to store the model)

Solution: Reduce the number of parameters

- Smaller per-trained model?

Pre-train a large model, but use a smaller model for the downstream tasks

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-21 16.14.15.png" alt="截屏2022-04-21 16.14.15" style="zoom:33%;" />



Share the parameters among the transformer layers

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 14.52.40.png" alt="截屏2022-04-22 14.52.40" style="zoom:33%;" />

##### Parameter-Efficient Fine-tuning

Use a small amount of parameters for each downsteam task

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 14.54.44.png" alt="截屏2022-04-22 14.54.44" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 14.54.55.png" alt="截屏2022-04-22 14.54.55" style="zoom:33%;" />

What is standard fine-tuning really doing?

- Modify the hidden representations(h) of the PLM such that it can perform well on downstream task

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 14.56.17.png" alt="截屏2022-04-22 14.56.17" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 14.56.46.png" alt="截屏2022-04-22 14.56.46" style="zoom:33%;" />

Fine-tuning = modifying the hidden representation based on a PLM

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 14.57.41.png" alt="截屏2022-04-22 14.57.41" style="zoom:33%;" />

###### Adapter

Use special submodules to modify hidden representations

Adapters: small trainable submodules inserted in transformers

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.00.24.png" alt="截屏2022-04-22 15.00.24" style="zoom:33%;" />

Inside of the transformer layer, only adapters are updated

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.09.56.png" alt="截屏2022-04-22 15.09.56" style="zoom:33%;" />

During fine-tuning, only update the adapters and the classified head

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.11.33.png" alt="截屏2022-04-22 15.11.33" style="zoom:33%;" />

###### LoRA

Use special submodules to modify hidden representations 

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.13.36.png" alt="截屏2022-04-22 15.13.36" style="zoom:33%;" />

LoRA: Low-Rank Adaptation of Large Language Models

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.14.16.png" alt="截屏2022-04-22 15.14.16" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.15.22.png" alt="截屏2022-04-22 15.15.22" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.20.26.png" alt="截屏2022-04-22 15.20.26" style="zoom:33%;" />

All downstream tasks share the PLM; the Lora in each layer and the classifier heads are the task-specific modules

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.21.52.png" alt="截屏2022-04-22 15.21.52" style="zoom:33%;" /> 

###### Prefix Tuning

Use special submodules to modify hidden representations

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.23.28.png" alt="截屏2022-04-22 15.23.28" style="zoom:33%;" />

What is prefix?

a letter or group of letters added to the beginning of a word wo make a new word

Standard Self-Attention

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.26.21.png" alt="截屏2022-04-22 15.26.21" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.28.19.png" alt="截屏2022-04-22 15.28.19" style="zoom:33%;" />

Prefix tuning: Only the prefix (key and value) are updated during fine-tuning

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.29.06.png" alt="截屏2022-04-22 15.29.06" style="zoom:33%;" />

###### Soft Prompting

Perpend the prefix embedding at the input layer

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.30.08.png" alt="截屏2022-04-22 15.30.08" style="zoom:33%;" />

Soft Prompting can be considered as the soften version of prompting

- (Hard) prompting: add words in the input sentence (fine-tune the model while fixing the prompts)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.31.44.png" alt="截屏2022-04-22 15.31.44" style="zoom:33%;" />

Soft Prompts: vectors (can be initialized from some word embeddings)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.33.08.png" alt="截屏2022-04-22 15.33.08" style="zoom:33%;" />

Hard Prompts: words (that are originally in the vocabulary)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.33.16.png" alt="截屏2022-04-22 15.33.16" style="zoom:33%;" />

**Benefit 1**: Drastically decreases the task-specific parameters

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.34.21.png" alt="截屏2022-04-22 15.34.21" style="zoom:33%;" />

**Benefit 2**: Less easier to overfit on training data; better out-of-domain performance 

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.37.11.png" alt="截屏2022-04-22 15.37.11" style="zoom:33%;" />

**Benefit 3**: Fewer parameters to fine-tune; a good candidate when training with small dataset

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.37.50.png" alt="截屏2022-04-22 15.37.50" style="zoom:33%;" />

##### Early Exit

Problem 1: the PLM is too big

- Inference takes too long

Inference using the whole model takes too long

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.39.36.png" alt="截屏2022-04-22 15.39.36" style="zoom:33%;" />

Simpler data may require lesser effort to botain the answer 

Reduce the number of layers used during inference

Add a classifier at each layer

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.41.09.png" alt="截屏2022-04-22 15.41.09" style="zoom:33%;" />

How do we know which classifier to ues?

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.42.04.png" alt="截屏2022-04-22 15.42.04" style="zoom:33%;" />

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.43.04.png" alt="截屏2022-04-22 15.43.04" style="zoom:33%;" />

Early exit reduces the inference time while keeping the performance 

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 15.43.41.png" alt="截屏2022-04-22 15.43.41" style="zoom:33%;" />

##### Summary

Parameter-efficient fine-tuning: Reduce the task-specific parameters in downstream task

Early exit: Reduce the models that are involved during inference

### Closing Remarks

What we address in this lecture

- Making PLM smaller, faster, and more parameter-efficient
- Deploying PLMs when the labeled data in the downstream task is scarce

The problems are not completely solved yed

The problems we discuss are just a small part of problems of PLMs

- Why does self-supervised per-training work
- Interpertabilitu of the model's prediction
- Domain adaptation
- Continual learning / lifelong learning
- Security and privacy

## Self-supervised Learning for Speech and Image

### Review

#### Self-supervised Learning for Text

 <img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.02.37.png" alt="截屏2022-04-22 17.02.37" style="zoom:33%;" />

#### Self-supervised Learning for Speech

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.06.59.png" alt="截屏2022-04-22 17.06.59" style="zoom:33%;" />

**S**peech processing **U**niversal **PER**formance **B**enchmark (SUPERB)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.09.35.png" alt="截屏2022-04-22 17.09.35" style="zoom:33%;" />

#### Self-supervised Learning for Image

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.14.30.png" alt="截屏2022-04-22 17.14.30" style="zoom:33%;" />

### Generative Approaches

#### Masking

BERT series

How about speech? 1910.12638

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.17.13.png" alt="截屏2022-04-22 17.17.13" style="zoom:33%;" />

Smoothness of acoustic features 1910.12638

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.19.12.png" alt="截屏2022-04-22 17.19.12" style="zoom:33%;" />

Masking strategies for speech

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.19.56.png" alt="截屏2022-04-22 17.19.56" style="zoom:33%;" />

#### Predicting Future

GPT series

How about speech? 1910.12607

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.22.10.png" alt="截屏2022-04-22 17.22.10" style="zoom:33%;" />

### Predictive Approach

Speech and images contain many details that are difficult to generate.

Can a model learn without generation?

#### Image - Predicting Rotation

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.25.21.png" alt="截屏2022-04-22 17.25.21" style="zoom:33%;" />

#### Image - Context Prediction

1505.05192

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.26.07.png" alt="截屏2022-04-22 17.26.07" style="zoom:33%;" />

Similar idea on Speech

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.26.56.png" alt="截屏2022-04-22 17.26.56" style="zoom:33%;" />

#### Predict Simplified Objects

**Speech** HuBERT 2106.07447 BEST-RQ 2202.01855

**Image** DeepCluster 1807.05520

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-22 17.30.03.png" alt="截屏2022-04-22 17.30.03" style="zoom:33%;" />

### Contrastive Learning

Speech and images contain many details that are difficult to generate.

Can a model learn without generation? 

#### Basic Idea of Constrastive Learning

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.32.07.png" alt="截屏2022-04-23 14.32.07" style="zoom:33%;" />

#### SimCLR

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.35.58.png" alt="截屏2022-04-23 14.35.58" style="zoom:33%;" />

#### MoCo

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.37.39.png" alt="截屏2022-04-23 14.37.39" style="zoom:33%;" />

#### Contrastive Learning for Speech

 CPC 1807.03748

Wav2vec 1904.05862

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.53.14.png" alt="截屏2022-04-23 14.53.14" style="zoom:33%;" />

VQ-wav2vec 1910.05453

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.56.48.png" alt="截屏2022-04-23 14.56.48" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.57.17.png" alt="截屏2022-04-23 14.57.17" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.58.36.png" alt="截屏2022-04-23 14.58.36" style="zoom:33%;" />

Wav2vec 2.0

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 14.59.56.png" alt="截屏2022-04-23 14.59.56" style="zoom:33%;" />

Continuous input is critical

Quantized target improves performance

### Bootstrapping Approaches 

Learning without negative examples

#### Alterative way to understand Boostrapping

Typical Knowledge DIstillation

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 15.07.15.png" alt="截屏2022-04-23 15.07.15" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 15.07.47.png" alt="截屏2022-04-23 15.07.47" style="zoom:33%;" />

### Concluding Remarks

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-23 15.08.38.png" alt="截屏2022-04-23 15.08.38" style="zoom:33%;" />
