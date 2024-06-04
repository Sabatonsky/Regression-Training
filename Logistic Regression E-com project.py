# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 20:43:39 2023

@author: Bannikov Maxim
"""

import numpy as np
import pandas as pd

d = r"C:\Users\AMD\Documents\GitHub\machine_learning_examples\ann_logistic_extra\ecommerce_data.csv"

data = pd.read_csv(d)

def get_data():
    df = pd.read_csv(d)
    df = df.values
    
    X = df[:,:-1]
    Y = df[:, -1]
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
    
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)]
        
    Z = np.zeros((N,4))
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    X2[:,-4:] = Z
    
    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

X,Y = get_data()

D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, Y, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, Y, b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("score", classification_rate(Y, predictions))
