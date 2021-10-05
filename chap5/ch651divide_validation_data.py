import sys, os
sys.path.append(os.pardir)
import numpy as np

## 读入mnist数据
(x_train, t_train), (x_test, t_test) = load_mnist()

## 打乱训练数据
x_train, t_train = shuffle_dataset(x_train, t_train)	#此处是利用了np.random.shuffle函数实现的打乱数据，方法在common/util.py里

## 分隔validation data
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)

x_val = x_train[:validation_num]
t_val = x_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]