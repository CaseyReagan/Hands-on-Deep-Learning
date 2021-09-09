import sys, os 
sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pylab as plt

def numerical_centrol_diff(f,x):	# 中心差分算斜率（梯度）
	h = 1e-4
	return (f(x+h)-f(x-h)) / (2*h)

def function_2(x):					# 这是一个有两个变量的数学函数，是我们的目标函数 y = x0*x0 + x1*x1
	if x.ndim == 1:
		#return x[0]**2 + x[1]**2
		return np.sum(x**2)
	else:
		return np.sum(x**2,axis=1)

def numerical_grardient_1d(f,x):
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

