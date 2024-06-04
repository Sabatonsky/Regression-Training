# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 09:29:14 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_oughter = 10
R1 = np.random.randn(int(N/2)) + R_inner
theta = 2*np.pi*np.random.random(int(N/2))
X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]]).T

R2 = np.random.randn(int(N/2)) + R_oughter
X_oughter = np.concatenate([[R2*np.cos(theta)],[R2*np.sin(theta)]]).T


X = np.concatenate([X_inner, X_oughter])
T = np.array([0]*int(N/2) + [1]*int(N/2))
 
plt.scatter(X[:,0], X[:,1], c = T)
plt.show()

ones = np.array([[1]*N]).T

r = np.zeros((N,1))

for i in range(N):
    r[i] = np.sqrt(X[i,:].dot(X[i,:]))

Xb = np.concatenate((ones, r, X), axis = 1)
w = np.random.randn(D+2)

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
Y = sigmoid(Xb.dot(w))
    
def cross_entropy(T, Y):
    E = 0
    for t in range(N):
        if T[t] == 1:
            E -= np.log(Y[t])
        else:
            E -= np.log(1 - Y[t])
        return E
    
lr = 0.0001
error = []
l2 = 0.01

for i in range(5000):
    w -= lr*(Xb.T.dot(Y-T) - l2*w)
    Y = sigmoid(Xb.dot(w))
    e = cross_entropy(T, Y)    
    error.append(e)
    
plt.plot(error)
plt.scatter(range(N), r, c = T)
plt.plot(range(N), np.sort(Y))
plt.show()




    
    
    
     