class SGD(object):
	"""docstring for SGD"""
	def __init__(self, lr=0.01):
		self.lr = lr

	def update(self, params, grads):
		for key in params.keys()
			params[key] -= self.lr * grads[key]

"""
一段伪代码展示使用这个类的方法

network = TwoLayerNet(...)
optimizer = SGD()

for i in range(10000)
	...
	x_batch, t_batch =get_mini_batch(...) # mini-batch
	grads = network.gradient(x_batch, t_batch)
	params = network.params
	optimizer.update(params, grads)

这里optimizer意味着优化者
"""
