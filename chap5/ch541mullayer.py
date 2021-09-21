import sys, os
import numpy as np

class 	MulLayer(object):
	"""docstring for 	MulLayer"""
	def __init__(self):
		self.x = None
		self.y = None
	
	def forward(self, x, y):
		self.x = x
		self.y = y
		out = x * y

		return out

	def backward(self, dout):
		dx = dout * self.y   #翻转x和y
		dy = dout * self.x

		return dx,dy

	