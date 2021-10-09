import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col

class Convolution(object):
	"""docstring for Convolution"""
	def __init__(self, W, b, stride=1, pad=0):	##	初始化方法将卷积核的权重，偏置，步长，填充作为参数接收
		self.W = W
		self.b = b
		self.stride = stride
		self.pad = pad

	def forward(self, x):
		FN, C, FH, FW = self.W.shape 	##	卷积核是4维数据 卷积核数量、通道个数、高、宽
		N, C, H, W = x.shape
		out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
		out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

		col = im2col(x, FH, FW, self.stride, self.pad)		## 根据卷积核大小来展开输入数据
		col_W = self.W.reshape(FN, -1).T 	# 卷积核的展开为2维的数据,FN为卷积核的个数，reshape第二个参数设置为-1，是一种
											# 自动方法，函数会自动计算FN维度上的元素个数，比如原数据有750个数据，reshape(
											# 10,-1)之后就会转换成（10,75）形状的数据。最后做一个转置是因为卷积核展开后
											# 得是一个一列列的矩阵，才能和前面一行行的展开的输入数据矩阵相乘。
		out = np.dot(col, col_W) + self.b 	# 计算输入数据和卷积核的的矩阵的乘积

		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)	## transpose会更改多维数组的轴的顺序

		return out