# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 22:14:59 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

X = (np.random.random((N,D)) - 0.5)*10

true_w = np.array([1,0.5,-0.5] + [0]*(D-3))

Y = X.dot(true_w) + np.random.randn(N)*0.5

costs = []

w = np.random.randn(D) / np.sqrt(D)
r = 0.001
l1 = 10.0

for i in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - r*(X.T.dot(delta) + l1*np.sign(w))
    mse = delta.dot(delta) / N
    costs.append(mse)
    
plt.plot(costs)
plt.show()

plt.plot(true_w, label = 'true w')
plt.plot(w, label = 'map w')
plt.legend()
plt.show()