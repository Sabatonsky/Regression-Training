# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 15:26:11 2023

@author: Bannikov Maxim
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#X1 = systolic blood pressure
#X2 = age in years
#X3 = weight in pounds

d = r"C:\Users\AMD\Documents\GitHub\machine_learning_examples\linear_regression_class\mlr02.xls"

df = pd.read_excel(d, engine='xlrd')
X = df.values

plt.scatter(X[:,1],X[:,0])
plt.show()
plt.scatter(X[:,2],X[:,0]) 
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2','X3','ones']]
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r_2(X, Y):
    w = np.linalg.solve(X.T.dot(X),X.T.dot(Y))
    Yhat = X.dot(w)
    
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r_2 = 1 - d1.dot(d1)/d2.dot(d2)
    return r_2

X2_r = get_r_2(X2only, Y)
X3_r = get_r_2(X3only, Y)
Un_r = get_r_2(X, Y)