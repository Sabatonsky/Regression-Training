# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 11:21:39 2023

@author: Maksim Bannikov
"""

import numpy as np
import matplotlib.pyplot as plt

d = r"C:\Users\AMD\Documents\GitHub\machine_learning_examples\linear_regression_class\data_poly.csv"

X = []
Y = []

for line in open(d):
    x, y = line.split(',')
    x = float(x)
    X.append([1,x,x*x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,Y))
Yhat = np.dot(X,w)

plt.scatter(X[:,1], Y) 
plt.plot(sorted(X[:,1]),sorted(Yhat))
plt.show()
