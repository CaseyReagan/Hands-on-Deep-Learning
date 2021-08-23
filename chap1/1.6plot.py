import numpy as np 
import matplotlib.pyplot as plt 

x1 = np.arange(0, 6, 0.1)
y1 = np.sin(x1)

#plt.plot(x1, y1)
#plt.show()

y2 = np.cos(x1)

plt.plot(x1, y1, label="sin")
plt.plot(x1, y2, linestyle = "--", label="cos")
plt.xlabel("x")
plt.xlabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()

from matplotlib.image import imread
img = imread('lena.png')
plt.imshow(img)

plt.show()