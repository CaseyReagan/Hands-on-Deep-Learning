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
	x_history = []

	for i in range(step_num):
		x_history.append( x.copy() )	#首先append是为了每次都添加多一个x的点，然后x.copy()是做了一个
										#x的浅拷贝，在python里直接赋值会变成引用，会跟着被改变，想要复制出
										#一个一模一样值，但是不受影响的变量，就必须用copy()函数
		grad = numerical_gradient(f,x)
		x -= lr * grad 		# lr is learning rate

	return x, np.array(x_history)

def function_2(x):
	return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
gard_update,grad_history = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
#print("gard_update: "gard_update)
#print("init_x: ",init_x)	## 为什么init_x的值会被改变，因为在gradient_descent函数中我写了x=init_x，
							## 这里就是x是init_x的一个引用，而不是浅拷贝，所以会init_x会跟着x变，注意是list会跟着变

init_x = np.array([-3.0, 4.0])
lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
#print("x: ",x)
#print("x_history: ",x_history)
#print("init_x: ",init_x)	

plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()