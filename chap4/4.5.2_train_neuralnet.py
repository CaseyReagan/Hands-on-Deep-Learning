import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from gradient_2_layers_net import TwolayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

#超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
network = TwolayerNet(input_size=784, hidden_size=50,output_size=10)

for i in range(iters_num):
	#	获取mini-batch
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]

	#	计算梯度
	grad = network.numerical_gradient(x_batch, t_batch)
	#	grad = network.gradient(x_batch,t_batch)	# 高速版

	for key in ('W1', 'b1', 'W2', 'b2'):
		network.params[key] -= learning_rate * grad[key]

	# 记录学习过程
	loss = network.loss(x_batch, t_batch)
	train_loss_list.append(loss)