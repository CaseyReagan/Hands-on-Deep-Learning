class Man:
	def __init__(self, name):
		self.names = name
		print("Initialized!")

	def hello(self):
		print("Hello " + self.names + "!")

	def goodbye(self):
		print("Good-bye " + self.names + "!")


m = Man("TOM")
m.hello()
m.goodbye()