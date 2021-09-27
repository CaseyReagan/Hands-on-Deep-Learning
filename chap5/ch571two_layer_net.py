import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet(object):
	"""docstring for TwoLayerNet"""
	## 输入数据大小，隐藏层数据大小，输出层数据大小，
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
		# 初始化权重
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)	# 高斯分布一个i*h大小的随机矩阵
		self.params['b1'] = np.zeros(hidden_size)										# 偏置都用0初始化
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

		# 生成层
		self.layers = OrderedDict()							# OrderedDict是一个有序字典，它可以记住向字典里添加元素的顺序
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

		self.lastLayer = SoftmaxWithLoss()


	def predict(self, x):
		for layer in self.layers.values():		#此处layers一共有3个value,返回的都是视图对象（ view objects），提供了字典实体的动态视图，	
			x = layer.forward(x)		#这就意味着字典改变，视图也会跟着变化。视图对象不是列表，不支持索引，可以使用 list() 来转换为列表。
										#调用了每一层的forward函数，计算出了两层神经网络的输出（不包含softmax的）
		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)		#做了softmax和交叉熵误差之后的损失函数结果

	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)			# argmax返回的是最大数的索引.argmax有一个参数axis,默认是0(列),表示第几维的最大值.
		if t.ndim != 1 : t = np.argmax(t, axis=1)
		accuracy = np.sum(y == t) / float(x.shape[0])

		return accuracy

	def numerical_gradient(self, x, t):		# 用数值微分来算梯度,各级参数关于损失函数的倒数
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads

	def gradient(self, x, t):
		# forward
		self.loss(x, t)

		# backward
		dout = 1
		dout = self.lastLayer.backward(dout)

		layers = list(self.layers.values())		#	把values()返回的视图格式强制转换为list格式
		layers.reverse()						#	反转列表的顺序，这样各层就倒过来了
		for layer in layers:
			dout = layer.backward(dout)

		# 设定
		grads = {}
		grads['W1'] = self.layers['Affine1'].dW 	#计算各层参数的梯度
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db

		return grads