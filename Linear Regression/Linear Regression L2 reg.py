# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 11:02:21 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt

N = 50
X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)
Y[-1] += 30
Y[-2] += 30

plt.scatter(X,Y)

X = np.vstack([np.ones(N), X]).T 

w_ml = np.linalg.solve(X.T.dot(X),X.T.dot(Y))
Y_hat_ml = X.dot(w_ml)

l2 = 1000.0

w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X),X.T.dot(Y))
Y_hat_map = X.dot(w_map)

plt.scatter(X[:,1],Y)
plt.plot(X[:,1],Y_hat_ml)
plt.plot(X[:,1],Y_hat_map)
plt.show()
