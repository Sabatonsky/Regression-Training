# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 23:16:56 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
    ])

T = np.array([0,1,1,0])

ones = np.array([[1]*N]).T

xy = np.matrix(X[:,0]*X[:,1]).T

Xb = np.array(np.concatenate((ones, xy, X), axis=1))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cross_entropy(T,Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

w = np.random.randn(D + 2) / np.sqrt(D)
z = Xb.dot(w)
Y = sigmoid(z)

lr = 0.001
l2 = 0.01
error = []

for i in range(50000):
    e = cross_entropy(T, Y)
    error.append(e)
    if e % 100 == 0:
        print(e)
    w += lr*(Xb.T.dot(T-Y) - l2*w)
    
    Y = sigmoid(Xb.dot(w))
    
plt.plot(error)
plt.title('cross-entropy per iteration')
plt.show()

print('final weights:', w)
print('final classification rate:', 1 - np.abs(T-np.round(Y)).sum() / N)