import sys, os
sys.path.append(os.pardir)
import numpy as np

class Dropout(object):
	"""docstring for Dropout"""
	def __init__(self, dropout_ratio=0.5):
		self.dropout_ratio = dropout_ratio
		self.mask = None

	def forward(self, x, train_flg=True):
		if train_flg:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio	# 这里是随机生成了一个和x一样大小的数组
			return x * self.mask

		else:
			return x * (1.0 - self.dropout_ratio)

	def backward(self, dout):
		return dout * self.mask
