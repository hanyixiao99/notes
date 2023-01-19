## Convolutional Neural Network

### Image Classification

All the images to be classified have the same size (100x100).

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 19.34.42.png" alt="截屏2022-04-16 19.34.42" style="zoom:33%;" />

## HW3

### Objective Image Classification

Solve image classification with CNN.

Improve the performance with data augmentations.

Understand popular image model techniques such as residual.

### Model Selection

Visit [torchvision.models](https://pytorch.org/vision/stable/models.html) for a list of model structures, or go to [timm](https://github.com/rwightman/pytorch-image-models) for the latest model structures.

Pretrained weights are not allowed, specifically set `pertrained=False` to ensure that the guideline is met.

### Data Augmentation

Modify the image data so non-identical inputs are given to the model each epoch, to pervent overfitting of the model.

Visit [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) for a list of choices and their corresponding effect. Diversity is encouraged! Usually, stacking multiple transformations leads to better results.

Coding: fill in `train_tfm` to gain this effect

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 20.40.15.png" alt="截屏2022-04-16 20.40.15" style="zoom:33%;" />

### Advanced Data Augmentation - [mixup](https://arxiv.org/pdf/1710.09412.pdf)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 20.41.09.png" alt="截屏2022-04-16 20.41.09" style="zoom:33%;" />

Coding:

- In your `torch.utils.Dataset`, `__getitem__()` needs to return an image that is the linear combination of two images.

- In your `torch.utils.Dataset`, `__getitem__()` needs to return a label that is a vector, to assign probabilities to each class.

- You need to explicitly code out the math formula of the cross entropy loss, as `CrossEntropyLoss` dose not support multiple labels.

### Test Time Augmentation

The sample code tests images using a deterministic "test transformation"

You may using the train transformation for a more diversified representation of the images, and predict with multiple variants of the test images.

Coding: You need to fill in `train_tfm`, change the augmentation method for test_dataset, and modify prediction code to gain this effect

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 20.48.23.png" alt="截屏2022-04-16 20.48.23" style="zoom:33%;" />

Usually, `test_tfm` will produce images that are more identifiable, so you can assign a larger weight to `test_tfm` results for better performance.

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-16 20.50.15.png" alt="截屏2022-04-16 20.50.15" style="zoom:33%;" />

Ex: Final Prediction = avg_train_tfm_pred * 0.5 + test_tfm_pred * 0.5

### Ensemble

- Average of logits or probability: Need to save verbose output, less ambiguous

- Voting: Easier to implement, need to break ties

Coding: basic math operations with numpy or torch