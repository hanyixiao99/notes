# 17 使用和购买GPU

```
!nvidia-smi
```

深度学习框架默认在CPU，需要指定在GPU

```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```

暂时Pass

