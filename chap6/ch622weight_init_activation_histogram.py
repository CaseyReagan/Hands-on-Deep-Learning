import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1/(1 + np.exp(-x))

x = np.random.randn(1000,100) # 1000个数据 高斯分布
node_num = 100					# 各隐藏层节点的数量
hidden_layer_size = 5			# 5层隐藏层
activations = {}				# 激活值的结果存在这里

for i in range(hidden_layer_size):
	if i != 0:
		x = activations[i-1]

#	w = np.random.randn(node_num, node_num) * 1 	#标准差为1的高斯分布
#	w = np.random.randn(node_num, node_num) * 0.01 	#标准差为0.01的高斯分布
	w = np.random.randn(node_num, node_num) / np.sqrt(node_num)		#用Xavier方法设置初始值，前一层姐的节点为n，则
																	#初始值使用标准差为1 / sqrt(n)的分布

	z = np.dot(x, w)
	a = sigmoid(z)
	activations[i] = a

for i, a in activations.items():
	plt.subplot(1, len(activations), i+1)
	plt.title(str(i+1) + "-layer")
	plt.hist(a.flatten(), 30, range=(0,1))
plt.show()
