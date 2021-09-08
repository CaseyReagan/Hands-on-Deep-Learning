import sys, os
sys.path.append(os.pardir)
import numpy as np

## mean squared error均方误差，指把输出y的所有维度和对应的label t的所有维度，
## 对应相差之后再做平方，然后再全部加起来，最后除以2，这样可以得到y和t的差别的一个量化结果
## 要注意的是此处参数y和t都得是numpy数组
def mean_squared_error(y,t):
	return (1/2)*np.sum((y-t)**2)

## cross entropy error交叉熵误差，因为t label经常是用one hot label表示法
## 所以实际上只算了t为1那一处的自然数对数，在这个函数中，y值越大（正确概率越大）
## log(y)越接近于0，损失函数也就越小
def cross_entropy_error(y,t):
	delta = 1e-7	# 这是为了loge(y)里y=0的时候log(0)会变成负无穷，所以delta设置为一个微小数字10^-7次方
	return -np.sum(t*np.log(y+delta))	# log函数是默认底为e的指数函数

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1,0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0]
y2 = [0.1,0.05, 0.1, 0, 0.05, 0.1, 0, 0.6, 0, 0]

# t = np.array(t)
# print(t.shape)

# loss 肯定越小越好
error1 = mean_squared_error(np.array(y),np.array(t))
print("error1: ", error1)
error2 = mean_squared_error(np.array(y2),np.array(t))
print("error2: ", error2)

print(' ')

error1 = cross_entropy_error(np.array(y),np.array(t))
print("error1: ", error1)
error2 = cross_entropy_error(np.array(y2),np.array(t))
print("error2: ", error2)