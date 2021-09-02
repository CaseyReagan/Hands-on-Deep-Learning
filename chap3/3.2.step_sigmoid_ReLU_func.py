import numpy as np
import matplotlib.pylab as plt

#阶跃函数的一种写法，y是一个bool类型的数组
#然后用astype把它强制类型转换为int类型
"""
def step_function(x):
	y = x>0
	return y.astype(np.int)
"""

#这个函数是上面函数写法的再简化
def step_function(x):
	return np.array(x>0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #指定Y轴范围
plt.show()

def sigmoid(x):
	return 1/(1+np.exp(-x))

y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #指定Y轴范围
plt.show()

def relu(x):
	return np.maximum(0, x)

y = relu(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1) #指定Y轴范围
plt.show()