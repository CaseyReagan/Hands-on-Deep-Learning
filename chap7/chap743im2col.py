import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col		## im2col 的意思是image to column 把包含批数量的4维的输入数据转换为2维矩阵以适合卷积核

x1 = np.random.rand(1, 3, 7, 7)		## 批大小为1，通道为3，数据大小7*7的数据
col1 = im2col(x1, 5, 5, stride=1, pad=0)	# im2col有5个参数，分别是 input_data由数据（数据量，通道，高，长）的4维数组构成的输入数据
											# filter_h	卷积核的高
											# filter_w	卷积核的长
											# stride	卷积核的步长
											# pad 		填充
print(col1.shape)	#(9, 75)		## 卷积核的通道个数必须和输入数据一样为3，卷积核大小为每个25，所以展开的输入数据的第2维是75
									## 因为输入数据每一层是7*7 应用于5*5的卷积核 就会有3*3个数据 所以第一维是9
x2 = np.random.rand(10, 3, 7, 7)	# 10个数据
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)	#(90, 75)		## 因为输入的批是10倍，所以最后保存的信息也是10倍，所以第1维是90