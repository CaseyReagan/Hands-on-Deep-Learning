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

## test
print(numerical_gradient_1d(function_2,np.array([3.0, 4.0])))

def numerical_gradient(f,x):
	if x.ndim == 1:
		return numerical_gradient_1d(f,x)
	else:
		grad = np.zeros_like(x)

		for idx, x1 in enumerate(x):	#enumerate函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标.例如此处idx=0的时候x1就等于x[0]
			grad[idx] = numerical_gradient_1d(f, x1)

		return grad

def tangent_line(f,x):
	d = numerical_gradient(f, x)
	print(d)
	y = f(x) - d*x
	return lambda t: d*t + y

## 这个相当于程序的入口，而Python则不同，它属于脚本语言，
## 	不像编译型语言那样先将程序编译成二进制再运行，而是动态的逐行解释运行。也就是从脚本第一行开始运行，没有统一的入口。
##  一个Python源码文件（.py）除了可以被直接运行外，还可以作为模块（也就是库），被其他.py文件导入。不管是直接运行还是被导入，
##  .py文件的最顶层代码都会被运行（Python用缩进来区分代码层次），而当一个.py文件作为模块被导入时，
##  我们可能不希望一部分代码被运行。那些在import之后不希望被运行的代码就放在if __name__ == '__main__':之下
if __name__ == '__main__':
	x0 = np.arange(-2, 2.5, 0.25)
	x1 = np.arange(-2, 2.5, 0.25)
	## numpy提供的numpy.meshgrid()函数可以让我们快速生成坐标矩阵 X X X， Y Y Y。
	## 这里x0和x1是给meshgrid函数的横纵坐标的一维向量，返回的X是一个二维矩阵，记录的是每一个点的横坐标
	## Y返回的是每一个点的纵坐标(从小到大的，所以矩阵每一行都是同一值)，可以用这个函数来快速描绘一个坐标矩阵的网格
	X, Y = np.meshgrid(x0, x1)

	#print(X)
	#print(Y)

	## flatten()函数用来把多维的数组降低为一维的数组（向量），注意这个函数只能用在numpy arrary数组上，直接用在python list是不行的
	X = X.flatten()
	Y = Y.flatten()

	print(X)
	print(Y)

	#print(np.array([X, Y]))
	grad = numerical_gradient(function_2, np.array([X, Y]))
	print(grad)

	plt.figure()
	## 画箭头图，后面两个参数的作用是箭头指向的方向和距离
	## 在这个例子中，因为梯度最低点在坐标中心，所以梯度向量的负值，就是指向中心点的梯度下降的方向
	plt.quiver(X, Y, -grad[0], -grad[1], angles="xy",color="#666666")
	plt.xlim([-2, 2])
	plt.ylim([-2, 2])
	plt.xlabel('x0')
	plt.ylabel('x1')
	plt.grid()
	plt.legend()
	plt.draw()
	plt.show()

