# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:48:45 2023

@author: Maksim Bannikov
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

d = r"C:\Users\AMD\Documents\GitHub\machine_learning_examples\linear_regression_class\data_2d.csv"

X = []
Y = []

for line in open(d):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])
    Y.append(float(y))
    
X = np.array(X)
Y = np.array(Y)

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

d1 = Y-Yhat
d2 = Y-Y.mean()
r_2 = 1 - d1.dot(d1)/d2.dot(d2)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:,0],X[:,1],Y)

x_0 = np.tile(np.arange(100), (100,1))
x_1 = np.tile(np.arange(100), (100,1)).T
y_0 = x_0*w[0]+x_1*w[1]+w[2]
print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(w[2], w[0], w[1]))

ax.plot_surface(x_0,x_1,y_0, alpha=0.5)
plt.show()
