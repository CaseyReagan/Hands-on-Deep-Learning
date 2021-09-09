import sys, os 
sys.path.append(os.pardir) 
import numpy as np
import matplotlib.pylab as plt

def numerical_diff_forward(f, x):			#向前差分
	h = 1e-4							# 0.0001 利用极小值，但不能产生舍入误差
	return (f(x+h)-f(x)) / h

def numerical_centrol_diff(f, x):			#中心差分减小误差
	h = 1e-4
	return (f(x+h)-f(x-h)) / (2*h)

def function_1(x):						#目标函数
	return 0.01*x**2 + 0.1*x

def tangent_line(f, x):					# d是函数在x点的斜率
    d = numerical_centrol_diff(f, x)
    print(d)
    y = f(x) - d*x               		# y是函数与y轴的交点，也就是垂直位移，也就是算的截距
    print(y)		# 因为y = kx + b, 所以当我们知道一个点和它的斜率的时候，就可以y - kx = b获得截距
    return lambda t: d*t + y 			# 切线函数，t为参量，相当于返回了一个y = kx + b
## 这里的lambda函数又叫匿名函数，或者成为表达式，lambda函数减少了代码的冗余，同时可读性更高
## 此处它的语法就是lambda t：这里t就是这个函数的参数,冒号后面跟的就是函数的内容（省去给函数命名了）

## 绘制目标函数
x = np.arange(0, 20, 0.1)
y = function_1(x)

## 函数在5这一处的切线
tf = tangent_line(function_1, 5)
y2 = tf(x)

## 函数在10这一处的切线
tf2 = tangent_line(function_1, 10)
y3 = tf2(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title('y=0.01x^2+0.1x')
plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()
