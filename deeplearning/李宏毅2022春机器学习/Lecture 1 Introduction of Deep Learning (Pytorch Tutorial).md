## Pytorch Tutorial 1

https://www.bilibili.com/video/BV1Wv411h7kN?p=6

```python 
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size, shuffle=True) # Training:True Testing:False
```

### Dataset & Dataloader

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, file):
        # Read data & preprocress
        self.data = ...

    def __getitem__(self, index):
        # Returns one sample at a time
        return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
```

```python
dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.07.25.png" alt="截屏2022-04-15 16.07.25" style="zoom:33%;" />

### Tensors

High-dimensional matrices(arrays)

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.09.09.png" alt="截屏2022-04-15 16.09.09" style="zoom:33%;" />

#### Shape of Tensors

Check with ==.shape()==

<!--dim in PyTorch == axis in Numpy-->

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.11.47.png" alt="截屏2022-04-15 16.11.47" style="zoom:33%;" />



#### Creating Tensors

Directly from data (list or numpy.ndarray)

```python
x = torch.tensor([1, -1], [-1, 1])
x = torch.from_numpy(np.array([[1, -1], [-1, 1]]))
```

Tensor of constant zeros & ones

```python
x = torch.zeros([2, 2])
x = torch.ones([1, 2, 5])
```

#### Common Operations

Common arithmetic functions are supported, such as:

```python
z = x + y
z = x - y
y = x.sum()
y = x.mean()
y = x.pow(2)
```

==Transpose==: transpose two specified dimensions

```python
x = torch.zeros([2, 3]) # torch.Size([2, 3])
x = x.transpose(0, 1) # torch.Size([3, 2])
```

==Squeeze==: remove the specified dimension with length = 1

```python
x = torch.zeros([1, 2, 3]) # torch.Size([1, 2, 3])
x = x.squeeze(0) # torch.Size([2, 3])
```

==Unsqueeze==: expand a new dimension

```python
x = torch.zeros([2, 3]) # torch.Size([2, 3])
x = x.unsqueeze(1) # torch.Size([2, 1, 3])
```

==Cat==: concatenate multiple tensors

```python
x = torch.zeros([2, 1, 3])
y = torch.zeros([2, 3, 3])
z = torch.zeros([2, 2, 3])
w = torch.cat([x, y, z], dim=1) # torch.Size([2, 6, 3])
```

#### Data Type

Using different data types for model and data will cause errors

#### Device

Tensors & modules will be computed with CPU by default

Use ==.to()== to move tensors to appropriate devices

```python
x = x.to('cpu')
x = x.to('cuda')
```

Check if your computer has NVIDIA GPU

```python
torch.cuda.is_available()
```

#### Gradient Calculation

```python
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z = x.pow(2).sum()
z.backward()
x.grad
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.33.47.png" alt="截屏2022-04-15 16.33.47" style="zoom:33%;" />

### Training & Testing Neural Networks - in PyTorch

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.35.12.png" alt="截屏2022-04-15 16.35.12" style="zoom:33%;" />

#### torch.nn - Neural Network Layers

Linear Layer (Fully-connected Layer)

```python
nn.Linear(in_features, out_features)
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.37.10.png" alt="截屏2022-04-15 16.37.10" style="zoom:33%;" />

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.39.23.png" alt="截屏2022-04-15 16.39.23" style="zoom:33%;" />

```python
layer = torch.nn.Linear(32, 64)
layer.weight.shape # torch.Size([64, 32])
layer.bias.shape # torch.Size([64])
```

#### torch.nn - Non-Linear Activation Functions

```python
nn.Sigmoid()
nn.ReLU()
```

#### torch.nn - Build your own neural network

```python
import torch.nn as nn

class MyModel(nn.module):
  def __init__(self):
    # Initialize your model & define layers
    super(MyModel, self).__init__()
    self.net = nn.Sequential(
    		nn.Linear(10, 32),
    		nn.Sigmoid(),
    		nn.Linear(32, 1)
    )
    # self.layer1 = nn.Linear(10, 32)
    # self.layer2 = nn.Sigmoid()
    # self.layer3 = nn.Linear(32, 1)
    
  def forward(self, x):
    # Compute output of your NN
    return self.net(x)
  	# out = self.layer1(x)
    # out = self.layer2(out)
    # out = self.layer3(out)
    # return out
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.50.56.png" alt="截屏2022-04-15 16.50.56" style="zoom:33%;" />

#### torch.nn - Loss Functions

```python
criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

loss = criterion(model_output, expected_value)
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 16.54.09.png" alt="截屏2022-04-15 16.54.09" style="zoom:33%;" />

#### torch.optim

Gradient-based optimization algorithms that adjust network parameters to reduce error

E.g. Stochastic Gradient Descent (SGD)

```python
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0)
```

For every batch of data:

1. Call `optimizer.zero_grad()` to reset gradients of model parameters

2. Call `loss.backward()` to back propagate gradients of prediction loss

3. Call `optimizer.step()` to adjust model parameters

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 17.06.20.png" alt="截屏2022-04-15 17.06.20" style="zoom:33%;" />

 #### Neural Network Training Setup

```python
dataset = MyDataset(file) # read data via MyDataset
tr_set = DataLoader(dataset, 16, shuffle=True) # put dataset into Dataloader
model = MyModel().to(device) # construct model and move to device(cpu/cuda)
criterion = nn.MSELoss() # set loss function
optimizer = torch.optim.SGD(model.parameters(), 0.1) # set oprimizer
```

#### Nerual Network Training Loop

```python
for epoch in range(n_epochs): # iterate n epochs
  model.train() # set model to train mode
  for x, y in tr_set: # iterate through the dataloader
    optimizer.zero_grad() # set gradient to zero
    x, y = x.to(device), y.to(device) # move data to device
    pred = model(x) # forward pass (compute output)
    loss = criterion(pred, y) # compute loss
    loss.backward() # compute gradient (back propagation)
    optimizer.step() # update model with optimizer
```

#### Neural Network Validation Loop

```python
model.eval() # set model to evaluation mode
total_loss = 0
for x, y in dv_set: # iterate through the dataloader
  x, y = x.to(device), y.to(device) # move data to device
  with torch.no_grad(): # disable gradient calculation
    pred = model(x) # forward pass (compute output) 
    loss = criterion(pred, y) # compute loss
  total_loss += loss.cpu().item() * len(x) # accumulate loss
  avg_loss = total_loss / len(dv_set.dataset) # compute averaged loss
```

#### Neural Network Testing Loop

```python
model.eval()
preds = []
for x in tt_set:
  x = x.to(device)
  with torch.no_grad():
    pred = model(x)
    preds.append(pred.cpu())
```

### Save/Load Trained Models

```python
torch.save(model.state_dict(), path # save
           
ckpt = torch.load(path) # load
model.load_state_dict(ckpt)
```

## PyTorch Tutorial 2

https://www.bilibili.com/video/BV1Wv411h7kN?p=7

### Task Description

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 20.11.52.png" alt="截屏2022-04-15 20.11.52" style="zoom:33%;" />

### Data

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2022-04-15 20.12.57.png" alt="截屏2022-04-15 20.12.57" style="zoom:33%;" />

### Load data & Preprocessing

```python
train_data = pd.read_csv('./covid.train.csv').drop(columns=['date']).values

x_train, y_train = train_data[:, :-1], train_data[:, -1]
```

 ### Dataset

```python
class COVID19Dataset(Dataset):
  '''
  x: Features
  y: Targets, if none, do prediction
  '''
  def __init__(self, x, y=None):
    # Read data and preprocess.
    if y is None:
      self.y = y
    else:
      self.y = torch.FloatTensor(y)
    self.x = torch.FloatTensor(x)
    
  def __getitem__(self, idx):
    # Return one sample at a time.
    # In this case, one sample includes a 117 dimensional feature and a label.
    if self.y is None:
      return self.x[idx]
    else:
      return self.x[idx], self.y[idx]
  
  def __len__(self):
    # Return the size of the dataset.
    # In this case, it's 2699.
    return len(self.x)
```

```python
train_dataset = COVID19Dataset(x_train, y_train)
```

### Dataloader

Group data into batches

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
```

If you set `shuffle=True`,dataloader will permutes the indices of all samples automatically. We often set this during Training.

### Model

```python
class My_model(nn.Module):
  def __init__(self, input_dim):
    # The input dimension of this model will be 117.
    super(My_model, self).__init__()
    # TODO: modify model's structure, be aware of dimension.
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 64),
    	nn.ReLU(),
    	nn.Linear(64, 32),
    	nn.ReLU(),
    	nn.Linear(32, 1)
    )
  
  def forward(self, x):
    x = self.layers(x)
    x = x.squeeze(1) # (B, 1) -> (B)
    return x
```

```python
model = My_model(input_dim=x_train.shape[1]).to('cuda')
# The output of this model will be a scalar, which represents the predicting value of the percentage of new tested positive cases in the 5th day.
```

### Criterion

We are doing a regression task, choosing mean square error as our loss function would be a good idea.

```python
criterion = torch.nn.MSELoss(reduction='mean')
```

### Optimizer

We need to declare a optimizer that adjust network parameters in order to reduce error.

Here we choose stochastic gradient descent as our optimization algorithm.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
```

### Training Loop

Get model prediction, compute gradient, update parameters and reset the gradient of model parameters.

```python
for epoch in range(3000):
  model.train() # set model to train mode
  train_pbar = tqdm(train_loader, position=0, leave=True)
  # tqdm is a package to visualize your training progress.
  # from tqdm import tqdm
  
  for x, y in train_pbar:
    optimizer.zero_grad() # set gradient to zero
    x, y = x.to('cuda'), y.to('cuda')
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## HW1

 https://github.com/virginiakm1988/ML2022-Spring/blob/main/HW01/HW01.ipynb

### Some utility Functions

Fixes random number generator seeds for reproducibility.

```python
def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Split provided training data into training set and validation set.

```python
def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
```

Predict.

```python
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds
```

### Feature Selection

Choose features you deem useful by modifying the function below.

```python
def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:,-1], valid_data[:,-1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:,:-1], valid_data[:,:-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0,1,2,3,4] # TODO: Select suitable feature columns.
        
    return raw_x_train[:,feat_idx], raw_x_valid[:,feat_idx], raw_x_test[:,feat_idx], y_train, y_valid
```

### Training Loop

```python
def trainer(train_loader, valid_loader, model, config, device):

    criterion = nn.MSELoss(reduction='mean') # Define your loss function, do not modify this.

    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 

    writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
```

### Configurations

`config` contains hyper-parameters for training and the path to save your model.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.            
    'batch_size': 256, 
    'learning_rate': 1e-5,              
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
```

### Dataloader

Read data from files and set up training, validation, and testing sets. You do not need to modify this part.

```python
# Set seed for reproducibility
same_seed(config['seed'])


# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days) 
# test_data size: 1078 x 117 (without last day's positive rate)
train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
```

### Strat training

```python
model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)
```

### Plot learning curves with `tensorboard`

`tensorboard` is a tool that allows you to visualize your training progress.

If this block does not display your learning curve, please wait for few minutes, and re-run this block. It might take some time to load your logging information.

```python
%reload_ext tensorboard
%tensorboard --logdir=./runs/
```

### Testing

The predictions of your model on testing set will be stored at `pred.csv`.

```python
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device) 
save_pred(preds, 'pred.csv')
```