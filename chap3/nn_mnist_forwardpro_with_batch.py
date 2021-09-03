import sys, os
sys.path.append(os.pardir)
import numpy as np

from dataset.mnist import load_mnist
import pickle
from common.functions import sigmoid, softmax

def get_data():
	(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False, one_hot_label = False)
	return x_test, t_test

def init_network():
	with open("sample_weight.pkl","rb") as f:
		network = pickle.load(f)

	return network

def predict(network, x):
	W1,W2,W3 = network['W1'],network['W2'],network['W3']
	b1,b2,b3 = network['b1'],network['b2'],network['b3']
	a1 = np.dot(x,W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1,W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2,W3) + b3
	y = softmax(a3)

	return y

x, t = get_data()
network = init_network()
W1,W2,W3 = network['W1'],network['W2'],network['W3']

#np.set_printoptions(threshold=np.inf)	#完整打印
print(x.shape)			# (10000, 784)
#print(x)
#print(x[0])
print(t.shape)			# (10000,)
print(t)
#print(network)
print(W1)
print(W1.shape)			# (784, 50)
print(W2.shape)			# (50, 100)
print(W3.shape)			# (100, 10)

# batch
batch_size = 100  # number of batch
accuracy_cnt = 0

for i in range(0,len(x),batch_size):	# len(x) = 10000, batch_size = 100 , i = 0, 100, 200 ... 9900 
	#print(i)
	x_batch = x[i:i+batch_size]			#  x_batch = x[0...99], x[100...199]
	y_batch = predict(network,x_batch)
	p = np.argmax(y_batch, axis=1) 		# argmax()用来获取最大的元素的索引，参数axis=1表示沿着第一维方向来找最大值，想当与按行找，axis=0是按列找
	accuracy_cnt += np.sum(p == t[i:i+batch_size])   # 这里的sum是把p列表里和t列表里，相同的元素的返回值True(1)加起来

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
