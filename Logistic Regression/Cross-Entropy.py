# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 23:50:33 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)

X[:50,:] = X[:50,:] - 2*np.ones((50,D))
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

T = np.array([0]*50 + [1]*50)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1)

w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T,Y):
    E=0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

r = 0.1
costs = []

for t in range(10000):
    z = Xb.dot(w)
    Y = sigmoid(z)
    delta = Y - T
    w = w - r/Xb.shape[0]*Xb.T.dot(delta)
    cee = cross_entropy(T,Y)
    costs.append(cee)
    
plt.plot(costs)

print(costs[-1])

w = np.array([0,4,4])

z = Xb.dot(w)
Y = sigmoid(z)
print(cross_entropy(T,Y))

plt.scatter(X[:,0], X[:,1], c=T, s=100, alpha = 0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
