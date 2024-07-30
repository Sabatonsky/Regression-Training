# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:01:15 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N,D)
X[50:,:] = X[50:,:] - 2*np.ones((50,D))
X[:50,:] = X[:50,:] + 2*np.ones((50,D))
T =  np.array([0]*50 + [1]*50)
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1)

w = np.random.randn(D + 1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

r = 0.1
costs = []
costs_2 = []

for t in range(10000):
    z = Xb.dot(w)
    Y = sigmoid(z)
    delta = Y - T
    w = w - r/N*Xb.T.dot(delta)
    cee = cross_entropy(T, Y)
    costs.append(cee)
    
print(costs[-1])
print("First w:", w)

lr = 0.1

for i in range(10000):
    
    Y = sigmoid(Xb.dot(w))
    w += lr * (np.dot((T-Y).T, Xb) - 0.1*w)    
    cee = cross_entropy(T, Y)
    costs_2.append(cee)

print(costs_2[-1])
print("Final w:", w)

plt.plot(costs)
plt.plot(costs_2)
plt.show()