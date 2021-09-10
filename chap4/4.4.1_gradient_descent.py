import sys, os
sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pylab as plt

## 一个简单的梯度下降的例子
def numerical_gradient_1d(f,x):
	h = 1e-4
	grad = np.zeros_like(x)  # 给梯度生成一个跟x大小相同的初始值为0的数组

	for idx in range(x.size):	# .size返回的是矩阵里元素的个数
		tem_val = x[idx]	 # 这是一个用来传参的变量
		x[idx] = tem_val + h 	# 此处给x的第idx个元素加了一个极小值
		fxh1 = f(x)			# 此处x的元素里有一个被加了极小值，另外一个没变，算此处偏导数的斜率

		x[idx] = tem_val - h 	# 此处是x的元素里有一个被减了极小值，另一个没变，算此处偏导数的斜率
		fxh2 = f(x)

		grad[idx] = (fxh1 - fxh2) / (2*h)	#利用中心差分算斜率
		x[idx] = tem_val	# 此处是斜率计算完毕，要把x的值还原

	return grad

def numerical_gradient(f,x):
	if x.ndim == 1:
		return numerical_gradient_1d(f,x)
	else:
		grad = np.zeros_like(x)

		for idx, x1 in enumerate(x):	#enumerate函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标.例如此处idx=0的时候x1就等于x[0]
			grad[idx] = numerical_gradient_1d(f, x1)

		return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
	x = init_x

	for i in range(step_num):
		grad = numerical_gradient(f,x)
		x -= lr * grad 		# lr is learning rate

	return x

def function_2(x):
	return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
gard_update = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(gard_update)

