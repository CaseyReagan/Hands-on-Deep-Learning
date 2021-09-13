import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from gradient_2_layers_net import TwolayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)
#iter_per_epoch = 10

#超参数
iters_num = 10000		# 梯度更新的次数，此处为10000次
#iters_num = 100
train_size = x_train.shape[0]	# 60000
batch_size = 100 		# mini batch的大小为100 ，这样相当于每次要从60000个训练数据中随机取出100个
learning_rate = 0.1
network = TwolayerNet(input_size=784, hidden_size=50,output_size=10)

for i in range(iters_num):
	#	获取mini-batch
	batch_mask = np.random.choice(train_size, batch_size)	# 从train_size这么多个数据（60000）中随机抽取batch_size这么多个数据(100)个
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	#	计算梯度
	grad = network.numerical_gradient(x_batch, t_batch)
	# grad = network.gradient(x_batch,t_batch)	# 高速版

	# 梯度下降，因为这里mini batch是随机选的，所以是SGD方法，随机梯度下降法
	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	# 记录学习过程，记录每一次的loss
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)

#	print("i: ",i)

	#计算每个epoch的识别精度,每600次循环为一个epoch
	if i % iter_per_epoch == 0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train acc, test acc | " + str(train_acc) + "," + str(test_acc))

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