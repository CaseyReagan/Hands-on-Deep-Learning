import sys, os
sys.path.append(os.pardir)
import numpy as np

def cross_entropy_error_with_onehotlabel(y,t):
	if y.ndim == 1:							# 这里是判断y是不是一维的，也就是判断是否有mini_batch
		y = y.reshape(1, t.size)			# 如果是一维的，那么就把输入的变量变为一维的np数组
		t = t.reshape(1, t.size)

	batch_size = y.shape[0]					# batch size为y的第一维的大小，要么是1维的，要么是y输入的mini_batch的个数
	return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error(y,t):
	if y.ndim == 1:
		y = y.reshape(1, t.size)
		t = t.reshape(1, t.size)

	batch_size = y.shape[0]
	return -np.sum(t * np.log(y[np.arange(batch_size)] + 1e-7)) / batch_size


t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0]

error = cross_entropy_error_with_onehotlabel(np.array(y),np.array(t))
print(error)

error2 = cross_entropy_error(np.array(y),np.array(t))
print(error2)