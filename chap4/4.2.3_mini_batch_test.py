import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train) , (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

## 采用mini batch的原因是，节约时间以及应对数据量过大的不现实情况，从全部数据中选取一部分，
## 作为全部数据的“近似数据”，这种选择最好是随机的，比如从60000个输入里，随机选择100个作为一个batch

train_size = x_train.shape[0]	# 训练数据的总大小
batch_size = 10		# 我们想要的batch大小

##	从train_size这么多个数字中(60000),随机选择batch_size这么多个数字（10个），返回值是一个np数组
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
