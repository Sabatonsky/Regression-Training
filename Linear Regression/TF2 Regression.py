# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 19:47:54 2023

@author: Bannikov Maxim
"""

import tensorflow as tf
from tensorflow import keras as ks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv"

data = pd.read_csv(url, header=None).values

X = data[:,0].reshape(-1, 1)
Y = np.log(data[:,1])

plt.scatter(X,Y)

X = X - X.mean()

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
    ])

lr = 0.001
momentum = 0.9

model.compile(optimizer=tf.keras.optimizers.SGD(lr, momentum), loss='mse')

def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001

scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(X, Y, epochs = 200, callbacks= [scheduler])

plt.plot(r.history['loss'], label='loss')

print(model.layers)
print(model.layers[0].get_weights())

Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

w, b = model.layers[0].get_weights()
X = X.reshape(-1, 1)
Yhat2 = (X.dot(w) + b).flatten()
np.allclose(Yhat, Yhat2)

model.save(r"C:\Users\AMD\Desktop\Training_code\linearclassifier.h5")
model = ks.models.load_model('linearclassifier.h5')
print(model.layers)
