# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:18:14 2023

@author: Bannikov Maxim
"""
import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
e = np.random.normal(20, 5, len(X))
 
d = r"C:\Users\AMD\Documents\GitHub\machine_learning_examples\linear_regression_class\data_1d.csv"

for line in open(d):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))
    
X = np.array(X) + e
Y = np.array(Y)

plt.scatter(X,Y)

n = len(X)
den = X.dot(X)-np.mean(X)*np.sum(X)
a = (X.dot(Y) - np.sum(X)*Y.mean())/den
b = (Y.mean()*X.dot(X) - X.mean()*X.dot(Y))/den
plt.plot(X, a*X + b)
plt.show()

Y_hat = a*X + b
rss = np.sum((Y-Y_hat)**2)
tss = np.sum((Y-Y.mean())**2)
r_2 = 1 - rss/tss

d = r"C:\Users\AMD\Documents\GitHub\machine_learning_examples\linear_regression_class\moore.csv"

X = []
Y = []

non_decimal = re.compile(r'[^\d]+')
list_r = [line.split('\t')]

for line in open(d):
    r = line.split('\t')
    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)
    
X = np.array(X)
Y = np.array(Y)
    
Y = np.log(Y)

denominator = X.dot(X) - X.mean()*X.sum()
a = (X.dot(Y) - X.mean()*Y.sum())/denominator
b = (Y.mean()*X.dot(X) - X.mean()*X.dot(Y))/denominator

plt.scatter(X,Y)
plt.plot(X,a*X + b)
plt.show()

Y_hat = a*X + b
rss = np.sum((Y-Y_hat)**2)
tss = np.sum((Y-Y.mean())**2)
r_2 = 1 - rss/tss

d = r"C:\Users\AMD\Documents\GitHub\machine_learning_examples\linear_regression_class\data_2d.csv"

 
