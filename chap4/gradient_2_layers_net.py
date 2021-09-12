import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwolayerNet(object):
	"""docstring for TwolayerNet"""
	## 输入数据大小，隐藏层数据大小，输出层数据大小，
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)	# 高斯分布一个i*h大小的随机矩阵
		self.params['b1'] = np.zeros(hidden_size)		# 偏置都用0初始化
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	def predict(self, x):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'],	self.params['b2']

		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y

	def loss(self, x, t):
		y = self.predict(x)

		return cross_entropy_error(y, t)

	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)

		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy

	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])		#这里numerical_gradient依然是用的数值微分来做梯度
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])		#之后的章节会学习反向传播法来做更快的梯度计算
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads