import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
from scipy.signal import savgol_filter

fname = 'output.txt'

f = open(fname, 'r+')
lines = f.readlines()
train, val, batch = [], [], []
for l in lines:
	if 'Training Accuracy' in l:
		data = l.split(' ')
		batch.append(int(data[1]))
		train.append(float(data[4]))
	elif 'Validation Accuracy' in l:
		data = l.split(' ')
		val.append(float(data[4]))

batch = savgol_filter(batch[:-1], 51, 3)
train = savgol_filter(train[:-1], 51, 3)
val = savgol_filter(val, 51, 3)

print train
print '------'
print val
plt.plot(batch, train, 'r')
plt.plot(batch, val, 'b')
plt.ylim((0.4,1.0))
#plt.plot(x_3, y_increase3, 'o')
#plt.plot(x_3, y_decrease3, 'g')


plt.title("Orthographic 2D Representation Accuracy vs Batch Number")
plt.xlabel("Batch Number")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"], loc="lower right")
plt.show()