import sys, os
sys.path.append(os.pardir)
import numpy as np

def mean_squared_error(y,t):
	return (1/2)*np.sum((y-t)**2)

def cross_entropy_error(y,t):
	delta = 1e-7	# 这是为了loge(y)里y=0的时候log函数会出问题，所以delta设置为一个微小数字10^-7次方
	return -np.sum(t*np.log(y+delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0]

# t = np.array(t)
# print(t.shape)

error = mean_squared_error(np.array(y),np.array(t))
print(error)

error2 = cross_entropy_error(np.array(y),np.array(t))
print(error2)