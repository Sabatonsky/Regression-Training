# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:13:05 2023

@author: Bannikov Maxim
"""

import tensorflow as tf
from sklearn.datasets import load_breast_cancer

print(tf.__version__)

data = load_breast_cancer()

type(data) 

data.keys()

data.data.shape

data.target

data.target_names

data.target.shape

data.feature_names

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size = 0.33)
N, D = X_train.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(D,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(1, input_shape = (D,), activation = 'sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
r = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 100)

print("Train score:", model.evaluate(X_train, Y_train))
print("Test score:", model.evaluate(X_train, Y_train))

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

P = model.predict(X_test)
print(P)

import numpy as np
P = np.round(P).flatten()
print(P)

print("Manually calculated accuracy:", np.mean(P == Y_test))
print("Evaluate output:", model.evaluate(X_test, Y_test))
