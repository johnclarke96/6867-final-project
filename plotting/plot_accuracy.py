import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
from scipy.signal import savgol_filter

f = open('../network/output_9_multichannel_input.txt', 'r')
x_1 = []
x_2 = []
x_3 = []
y_train1 = []
y_increase1 = []
y_decrease1 = []
y_train2 = []
y_increase2 = []
y_decrease2 = []
y_train3 = []
y_increase3 = []
y_decrease3 = []
y_valid_1 = []
y_valid_2 = []
y_valid_3 = []

for line in f:
  if "Training Accuracy" in line:
    split_line = line.strip().split()
    iteration = int(split_line[1])
    accuracy = float(split_line[4])
    x_1.append(iteration)
    y_train1.append(accuracy)
  if "Validation Accuracy" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_valid_1.append(accuracy)
  if "Validation Increase" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_increase1.append(accuracy)
  if "Validation Decrease" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_decrease1.append(accuracy)

f = open('../network/output_5_resampled.txt', 'r')

for line in f:
  if "Training Accuracy" in line:
    split_line = line.strip().split()
    iteration = int(split_line[1])
    accuracy = float(split_line[4])
    x_2.append(iteration)
    y_train2.append(accuracy)
  if "Validation Accuracy" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_valid_2.append(accuracy)
  if "Validation Increase" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_increase2.append(accuracy)
  if "Validation Decrease" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_decrease2.append(accuracy)

'''
f = open('../network/output_5_resampled.txt', 'r')

for line in f:
  if "Training Accuracy" in line:
    split_line = line.strip().split()
    iteration = int(split_line[1])
    accuracy = float(split_line[4])
    x_3.append(iteration)
    y_train.append(accuracy)
  if "Validation Accuracy" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_valid_3.append(accuracy)
  if "Validation Increase" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_increase3.append(accuracy)
  if "Validation Decrease" in line:
    split_line = line.strip().split(':')
    accuracy = float(split_line[1])
    y_decrease3.append(accuracy)
'''
print max(y_train1)
print max(y_train2)


y_train1 = savgol_filter(y_train1, 51, 3)
y_valid_1 = savgol_filter(y_valid_1, 51, 3)
y_increase1 = savgol_filter(y_increase1, 51, 3)
y_decrease1 = savgol_filter(y_decrease1, 51, 3)


y_train2 = savgol_filter(y_train2, 51, 3)
y_valid_2 = savgol_filter(y_valid_2, 51, 3)
y_increase2 = savgol_filter(y_increase2, 51, 3)
y_decrease2 = savgol_filter(y_decrease2, 51, 3)

#y_train3 = savgol_filter(y_train3, 51, 3)
#y_valid_3 = savgol_filter(y_valid_3, 51, 3)
#y_increase3 = savgol_filter(y_increase3, 51, 3)
#y_decrease3 = savgol_filter(y_decrease3, 51, 3)

plt.plot(x_1, y_train1, 'r')
plt.plot(x_1, y_valid_1, 'b')
#plt.plot(x_1, y_increase1, 'r')
#plt.plot(x_1, y_decrease1, 'b')

plt.plot(x_2, y_train2, 'y')
plt.plot(x_2, y_valid_2, 'g')
#plt.plot(x_2, y_increase2, 'y')
#plt.plot(x_2, y_decrease2, 'g')

#plt.plot(x_3, y_train3, 'o')
#plt.plot(x_3, y_valid_3, 'g')
#plt.plot(x_3, y_increase3, 'o')
#plt.plot(x_3, y_decrease3, 'g')


plt.title("Single-Channel vs Multi-Channel Representation")
plt.xlabel("Batch Number")
plt.ylabel("Accuracy")
plt.legend(["Multi Training", "Multi Validation", "Single Training", "Single Validation"], loc="lower right")

plt.show()

