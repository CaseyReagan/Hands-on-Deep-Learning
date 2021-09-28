import numpy as np

class Momentum(object):
	"""doclr=0.01, momentum=0.9ing for Momentum"""
	def __init__(self, lr=0.01, momentum=0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None				#实例变量v会保存五题的速度。

	def update(self, params, grads):
		if self.v is None:
			self.v = {}				#第一次调用update的时候，v会以字典变量的形式保存 与参数结构相同的数据
			for key, val in params.items():
				self.v[key] = np.zeros_like(val)

		for key in params.keys():
			self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]	# v = a*v - lr * 倒数
			params[key] += self.v[key]											# W = W + v
