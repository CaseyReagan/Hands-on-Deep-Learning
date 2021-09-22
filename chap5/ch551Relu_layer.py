class Relu(object):
	"""docstring for Relu"""
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = (x <= 0) ##这里的mask存的是一组numpy数组，内容是True,False,只要输入的x内容<=0就会得到True
		out = x.copy()
		out[self.mask] = 0		## 小于等于0的位置上的元素会被替换为0

		return out

	def backward(self, dout):
		dout[self.mask] = 0		##	反向传播过来的值，在正向传播时小于等于0的地方的值会被赋值为0，其余地方按dout传过来的内容继续反向传播

		dx = dout

		return dx

		