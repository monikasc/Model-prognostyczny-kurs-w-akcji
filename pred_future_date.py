# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:21:08 2020

@author: TQGR38
"""
#%matplotlib qt 
from statistics import mean
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import regularizers
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import load_model


stock= pd.read_csv("msft_8days.csv", header=0)
x=stock.iloc[:,4]
x=x.to_numpy()
x=np.append(x,0)
model = load_model('msft_model.h5')
x = x.reshape((x.shape[0],1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(x)
X_scaled=scaled
size=X_scaled.shape[1]
size=x.shape[0]
X = np.reshape(X_scaled, (-1, 1, size))

predictions= model.predict(X)
prediction_reshaped = np.zeros((len(predictions), size+1))


prediction_r = np.reshape(predictions, (len(predictions),))


prediction_reshaped[:,size] = prediction_r


prediction_inversed = scaler.inverse_transform(prediction_reshaped)[:,size]
print(prediction_inversed)