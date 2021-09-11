import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simplenet(object):
	"""docstring for simplenet"""
	def __init__(self):
		self.W = np.random.randn(2,3) # 用高斯分布进行初始化，一个2*3的权重矩阵

	def predict(self, x):
		return np.dot(x, self.W)

	def loss(self, x, t):
		z = self.predict(x)
		y = softmax(z)
		loss = cross_entropy_error(y,t)

		return loss

net = simplenet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print("p: ",p)
y_label = np.argmax(p)	#最大值的索引 argmax返回的是最大数的索引
print("Maximum y_label: ",y_label)
t = np.array([1, 0, 0])
loss = net.loss(x,t)
print("The loss is: ",loss)

def f(W):		#这里的W是一个伪参数，因为numerical_gradient会在内部执行f(x)，为了与之兼容而定义了f(W)
	return net.loss(x,t)

dW = numerical_gradient(f, net.W)	# 求f这个函数关于net.W的各个参数的偏导数
print("gradient is: ",dW)

## 因为f是一个简单函数，所以可以用lambda表达式体现会更直观
#f2 = lambda w: net.loss(x,t)
#dW = numerical_gradient(f2,net.W)