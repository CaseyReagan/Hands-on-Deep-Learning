import numpy as np

class AdaGrad(object):
	"""docstring for AdaGrad"""
	def __init__(self, lr=0.01):
		self.lr = lr
		self.h = None		# h就是在adagrad方法里控制逐渐减小学习率的参数，他是之前所有梯度值的平方和

	def update(self, params, grads):
		if self.h is None:
			self.h = {}
			for key, val in params.items():
				self.h[key] = np.zeros_like(val)

		for key in params.keys():
			self.h[key] += grads[key] * grads[key]		# h存的是之前梯度值的平方和
			params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + le-7)		# W = W - lr * (1/sqrt(h) * dW)
																					# 加一个微小值的目的是防止分母为0