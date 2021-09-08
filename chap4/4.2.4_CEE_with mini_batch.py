import sys, os
sys.path.append(os.pardir)
import numpy as np

## 这个函数是CEE with onehotlabel的版本
def cross_entropy_error_with_onehotlabel(y,t):
	if y.ndim == 1:							# 这里是判断y是不是一维的，也就是判断是否有mini_batch
		y = y.reshape(1, y.size)			# 如果是一维的，那么就把输入的变量变为一维的np数组
		t = t.reshape(1, t.size)

	# 这是if语句以外的，如果y.ndim不是1维，也就是说如果是多维的batch，那这里就取得batch_size
	batch_size = y.shape[0]					# batch size为y的第一维的大小，要么是1维的，要么是y输入的mini_batch的个数
	return -np.sum(t * np.log(y + 1e-7)) / batch_size

## 这个函数是当t不是以one hot label表示的时候，比如标签是数字“2”，“7”，这个时候要做一个转换
def cross_entropy_error(y,t):
	if y.ndim == 1:
		y = y.reshape(1, y.size)
		t = t.reshape(1, t.size)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size
## 这里的要点是，虽然t label是一个值，所以就不能计算那些one hot label位置上为0的地方的自然数对数
## 所以我们其实只需要计算每一个batch上，t值对应的那个输出y值的交叉熵误差，
## 最终y[np.arange(batch_size),t]得到的数是y[0,2]; y[1,7]; y[n,t]这样的y的第n个batch里的第t个输出


t = [0,0,1,0,0,0,0,0,0,0]
t2 = [2]
y = [0.1,0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0]

error1 = cross_entropy_error_with_onehotlabel(np.array(y),np.array(t))
print('error1: ',error1)

error2 = cross_entropy_error(np.array(y),np.array(t2))
print('error2: ',error2)