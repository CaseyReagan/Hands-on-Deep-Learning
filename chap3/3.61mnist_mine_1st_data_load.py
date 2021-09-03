import sys, os
sys.path.append(os.pardir) #为了导入父目录中的文件
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, )
print(x_train.shape) # (10000, 784)
print(t_test.shape) # (10000,)