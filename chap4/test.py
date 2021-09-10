import sys, os
sys.path.append(os.pardir) 
import numpy as np

'''
def function(a_inside):
	b = a_inside

	b -= 2

	return b
'''

def function_2(a_inside):
	b = a_inside

	for i in range(200):
		b -= 0.01

	return b

def function_3(a_inside):
	b = a_inside

	b.append('12')

	return b

a_outside_numpy = np.array([-3.0, 4.0])
a_outside_python = 10
a_outside_python_list = ['10','11']

result_python = function_2(a_outside_python)
result_numpy = function_2(a_outside_numpy)
result_python_list = function_3(a_outside_python_list)

print("result_numpy: ",result_numpy)
print("a_outside_numpy: ",a_outside_numpy)
print("result_python: ",result_python)
print("a_outside_python: ",a_outside_python)
print("result_python_list: ",result_python_list)
print("a_outside_python_list: ",a_outside_python_list)