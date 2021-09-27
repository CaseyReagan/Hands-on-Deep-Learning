import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch571two_layer_net import TwoLayerNet

## 读入mnist数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label = True)

network = TwoLayerNet(input_size = 784, hidden_size=50, output_size=10)

## 取了前三行的数据
x_batch = x_train[:3]
t_batch = t_train[:3]

#print(x_batch[0])

#差分法算梯度
grad_numerical = network.numerical_gradient(x_batch, t_batch)
#反向传播法算梯度
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
	diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]))	# abs()函数的目的是返回绝对值 average函数是求平均
	print(key + ":" + str(diff))