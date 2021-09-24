import sys ,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *

class SoftmaxWithLoss(object):
	"""docstring for SoftmaxWithLoss"""
	def __init__(self):
		self.loss = None # 损失
		self.y = None	#softmax的输出
		self.t = None	#监督数据（one-hot vector)

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		dx = (self.y - self.t) / batch_size

		return dx

		