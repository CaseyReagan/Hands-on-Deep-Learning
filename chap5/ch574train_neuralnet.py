import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from ch571two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

## 读入mnist数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label = True)

network = TwoLayerNet(input_size = 784, hidden_size=50, output_size=10)

iters_num = 10000
#iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
#iter_per_epoch = 200

for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	# 通过误差反向传播求梯度
	grad = network.gradient(x_batch, t_batch)

	# 更新
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print(train_acc, test_acc)

# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
