import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.util import im2col, col2im

class SimpleConvNet(object):
	"""docstring for SimpleConvNet"""
	def __init__(self, input_dim=(1, 28, 28), 							## input_dim是输入数据的维度 通道 高 长
						conv_param={'filter_num':30, 'filter_size':5,	## 卷积层的超参数是一个字典 卷积核的数量30个
						'pad':0, 'stride':1},							## 卷积核大小 5*5
						hidden_size = 100,	output_size = 10,	## 隐藏层（全连接）的神经元数量 输出层神经元数量
						weight_init_std = 0.01 					## 初始化时权重的标准差
				):
		filter_num = conv_param['filter_num']
		filter_size = conv_param['filter_size']
		filter_pad = conv_param['pad']
		filter_stride = conv_param['stride']
		input_size = input_dim[1]								## 28 why?	因为28是输入数据的高
		conv_output_size = ((input_size - filter_size + 2*filter_pad) / filter_stride) + 1 		## 卷积输出的大小
		pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))	## 池化输出的大小

	##	权重的初始化
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
		self.params['b1'] = np.zeros(filter_num)								##	实际上这里的W1是卷积层的权重
		self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
		self.params['b2'] = np.zeros(hidden_size)							##	这里开始才是池化之后的全连接层权重
		self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b3'] = np.zeros(output_size)							##	第二层全连接层

	## 各层的实例
		self.layers = OrderedDict()
		self.layers['Conv1'] = Convolution(self.params['W1'],self.params['b1'],conv_param['stride'],conv_param['pad'])
		self.layers['Relu1'] = Relu()
		self.layers['Pool1'] = Pooling(pool_h=2, Pool_w=2, stride=2)	## 2*2的池化层 步长一般跟池化层高宽一至
		self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
		self.layers['Relu2'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
		self.last_layer = softmaxWithLoss()

	## 向前传播
	def predict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x

	## softmax
	def loss(self, x, t):
		y = self.predict(x)
		return self.last_layer.forward(y, t)

	