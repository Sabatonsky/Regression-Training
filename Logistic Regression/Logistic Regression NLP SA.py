# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 22:50:59 2023

@author: Bannikov Maxim
"""

import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup as bs

d = r"C:\Users\AMD\Downloads\sorted_data_acl\electronics\positive.review"

positive_reviews = bs(open(d).read(), features="lxml")
positive_reviews = positive_reviews.find_all('review_text')

d = r"C:\Users\AMD\Downloads\sorted_data_acl\electronics\negative.review"

negative_reviews = bs(open(d).read(), features="lxml")
negative_reviews = negative_reviews.find_all('review_text')

wordnet_lemmatizer = WordNetLemmatizer()

negative_matrix = np.zeros((1,1))
positive_matrix = np.zeros((1,1))

def tokenizer(s):
    s = s.lower()
    s = ''.join(i for i in s if i.isalpha() or i == " ")
    tokens = nltk.word_tokenize(s)
    tokens = [wordnet_lemmatizer.lemmatize(i) for i in tokens]
    tokens = [i for i in tokens if i not in stopwords.words('english')]
    tokens = [i for i in tokens if len(i) > 2]
    return tokens

def tokens_to_vector(tokens, label):
    pass

token_dict = {}
current_index = 0

for i in range(len(negative_reviews)):
    tokens = tokenizer(negative_reviews[i].text)
            
    for token in tokens:           
        if token not in token_dict:
            token_dict[token] = negative_matrix.shape[1]
            negative_matrix[i, token_dict[token]-1] += 1
            negative_matrix = np.c_[negative_matrix, np.zeros((negative_matrix.shape[0], 1))]            
        else:
            negative_matrix[i, token_dict[token]-1] += 1
            
        negative_matrix[i,:] = negative_matrix[i,:] / negative_matrix[i,:].sum() 
                
    negative_matrix = np.r_[negative_matrix, np.zeros((1,negative_matrix.shape[1]))]
     
negative_matrix = np.delete(negative_matrix, -1, axis=1)
negative_matrix = np.delete(negative_matrix, -1, axis=0)

positive_matrix = np.zeros((1,negative_matrix.shape[1] + 1))

for i in range(len(positive_reviews)):
    tokens = tokenizer(positive_reviews[i].text)
            
    for token in tokens:           
        if token not in token_dict:
            token_dict[token] = positive_matrix.shape[1]
            positive_matrix[i, token_dict[token]-1] += 1
            positive_matrix = np.c_[positive_matrix, np.zeros((positive_matrix.shape[0], 1))]            
        else:
            positive_matrix[i, token_dict[token]-1] += 1
            
        positive_matrix[i,:] = positive_matrix[i,:] / positive_matrix[i,:].sum()         
        
    positive_matrix = np.r_[positive_matrix, np.zeros((1,positive_matrix.shape[1]))]
     
positive_matrix = np.delete(positive_matrix, -1, axis=1)
positive_matrix = np.delete(positive_matrix, -1, axis=0)

negative_matrix = np.c_[negative_matrix, np.zeros((negative_matrix.shape[0], positive_matrix.shape[1] - negative_matrix.shape[1]))]     

N = negative_matrix.shape[0] + positive_matrix.shape[0]

ones = np.array([[1]*int(N/2)]).T
zeros = np.array([[0]*int(N/2)]).T

negative_matrix = np.concatenate((ones, negative_matrix, zeros), axis = 1)
positive_matrix = np.concatenate((ones, positive_matrix, ones), axis = 1)

data = np.concatenate((negative_matrix, positive_matrix), axis = 0)
data = shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

D = X.shape[1]
w = np.random.randn(D)/np.sqrt(D)

X_train = X[:-100,]
Y_train = Y[:-100,]

X_test = X[-100:,]
Y_test = Y[-100:,]

lr = 0.005
l1 = 0.5 * 0.001
l2 = (1-0.5)/2 * 0.001

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cross_entropy(Y, T):
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i]) + l2*(w**2).mean() + l1*np.abs(w).mean()
        else:
            E -= np.log(1-Y[i]) + l2*(w**2).mean() + l1*np.abs(w).mean()
    return E

costs_train = []

for i in range(5000):
    Y_hat = sigmoid(X_train.dot(w))
    w -= lr*(X_train.T.dot(Y_hat - Y_train) + 2*l2*w + l1*np.sign(w)) 
    cost = cross_entropy(Y_hat, Y_train)
    costs_train.append(cost)
    
plt.plot(costs_train)
costs_train[-1]

result = {i:[w[token_dict[i]]] for i in token_dict}
result_df = pd.DataFrame.from_dict(result, orient='index')

Y_result = sigmoid(X_test.dot(w))
cost = cross_entropy(Y_result, Y_test)

costs_train[-1]
print(cost)

false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0
    
for i in range(len(Y_result)):
    if Y_result[i] > 0.5:
        if Y_test[i] == 1:
            true_positive += 1
        else:
            false_positive += 1
    else:
        if Y_test[i] == 0:
            true_negative += 1
        else:
            false_negative += 1
            
positive = true_positive + false_positive
negative = true_negative + false_negative
            
accuracy = (true_positive + true_negative) / (positive + negative)

precision = true_positive / (true_positive + false_positive)

recall = true_positive / (true_positive + true_negative)

F_score = 2*(precision*recall)/(precision+recall)


