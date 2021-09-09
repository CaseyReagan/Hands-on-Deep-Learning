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
    y = f(x) - d*x               		# y是函数与y轴的交点，也就是垂直位移
    print(y)
    return lambda t: d*t + y 			# 切线函数，t为参量

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
