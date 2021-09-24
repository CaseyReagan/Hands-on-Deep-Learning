import sys ,os
sys.path.append(os.pardir)
import numpy as np

class Sigmoid(object):
	"""docstring for Sigmoid"""
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = 1 / (1 + np.exp(-x))
		self.out = out

		return out

	## 正向传播时将输出保存在了实例变量out中，然后反向传播的时候使用该变量out进行计算
	def backward(self, dout):
		dx = dout * (1 - self.out) * self.out
		
		return dx
