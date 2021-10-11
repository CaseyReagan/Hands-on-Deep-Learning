import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col, col2im

class Pooling(object):
	"""docstring for Pooling"""
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.pool_w = pool_w
		self.pool_h = pool_h
		self.stride = stride
		self.pad = pad

		self.x = None
		self.arg_max = None

	def forward(self, x):
		N, C, H, W = x.shape
		out_h = int(1 + (H - self.pool_h) / self.stride)	## 首先要计算出不转换原始数据的话池化之后的高和宽
		out_w = int(1 + (W - self.pool_w) / self.stride)

		## 展开（1）
		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)	## 根据池化核的大小展开
		col = col.reshape(-1, self.pool_h * self.pool_w)	## 把展开数据改变为h*w这么多行，-1维度自动计算出需要多少列

		arg_max = np.argmax(col, axis=1)	## 记录下最大值的坐标
		## 最大值（2）
		out = np.max(col, axis=1)		## 采用的是max池化
		## 转换
		out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)	## 最后要把数据转换回原本的格式

		self.x = x
		self.arg_max = arg_max

		return out

	def backward(self, dout):
		dout = dout.transpose(0, 2, 3, 1)

		pool_size = self.pool_h * self.pool_w
		dmax = np.zeros((dout.size, pool_size))
		dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
		dmax = dmax.reshape(dout.shape + (pool_size,))

		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

		return dx
