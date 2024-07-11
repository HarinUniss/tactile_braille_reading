import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

list = [1, 15, 16, 3, 5, 6, 7, 1]

# plot data
plt.figure()
plt.scatter(range(len(list)), list)
plt.show()