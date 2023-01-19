# 15 实战：Kaggle房价预测

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

使用pandas读入并处理数据

```python
train_data = pd.read_csv('/Users/hanyixiao/Documents/Code/Python/deeplearning/data/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('/Users/hanyixiao/Documents/Code/Python/deeplearning/data/house-prices-advanced-regression-techniques/test.csv')
print(train_data.shape)
print(test_data.shape)
# (1460, 81)
# (1459, 80)
```

前四个和最后两个特征，以及相应标签（房价）

```python
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
#    Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice
# 0   1          60       RL         65.0       WD        Normal     208500
# 1   2          20       RL         80.0       WD        Normal     181500
# 2   3          60       RL         68.0       WD        Normal     223500
# 3   4          70       RL         60.0       WD       Abnorml     140000
```

在每个样本中，第一个特征是ID，我们将其从数据集中删除

```python
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

将所有缺失的值替换为相应特征的平均值

通过将特征重新缩放到零均值和单位方差来标准化数据

```python
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

处理离散值

用一次独热编码替换他们

```python
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
# (2919, 331)
```

从pandas格式中提取numpy格式，并将其转换为张量表示

```python
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
```

训练

```python
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1)) # 单层线性回归
    return net
```

我们更关心相对误差，解决这个问题的一种方法是用价格预测的对数来衡量差异

```python
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

训练函数将借助Adam优化器，是一个比较平滑的SGD，好处是对学习率并不敏感，适用范围宽

```python
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

K折交叉验证

```python
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

返回训练和验证误差的平均值

```python
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

模型选择，要做的就是不断调下列参数，最后看验证集平均rmse

```python
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
# 折1，训练log rmse0.170315, 验证log rmse0.156714
# 折2，训练log rmse0.162597, 验证log rmse0.190524
# 折3，训练log rmse0.163757, 验证log rmse0.168524
# 折4，训练log rmse0.168332, 验证log rmse0.154662
# 折5，训练log rmse0.164607, 验证log rmse0.183559
# 5-折验证: 平均训练log rmse: 0.165921, 平均验证log rmse: 0.170796
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-08 16.54.44.png" alt="截屏2021-12-08 16.54.44" style="zoom:50%;" />

提交Kaggle预测，进行一次完整的训练

```python
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
# 训练log rmse：0.162121
```

<img src="/Users/hanyixiao/Library/Application Support/typora-user-images/截屏2021-12-08 16.57.41.png" alt="截屏2021-12-08 16.57.41" style="zoom:50%;" />