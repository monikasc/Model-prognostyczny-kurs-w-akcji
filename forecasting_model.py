# -*- coding: utf-8 -*-
"""
@author: Monika Scislo
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


stock= pd.read_csv("msft.csv", header=0)
stock.index=stock['Date']
stock=stock.drop(columns='Date')
df=stock
# =============================================================================
def generate_series(data, value_num):
    data=data.to_numpy()
    close = data[:,3]
    dividends = data[:,5]
    tsg = TimeseriesGenerator(close,close, 
                               length=value_num,
                               batch_size=len(close))
    global_index = value_num
    i, t = tsg[0]
    has_dividends = np.zeros(len(i)) 
    for b_row in range(len(t)):
        assert(abs(t[b_row] - close[global_index]) <= 0.001)
        has_dividends[b_row] = dividends[global_index] > 0            
        global_index += 1
    return np.concatenate((i, np.transpose([has_dividends])),
                           axis=1), t
                          

scaler_filename = "scaler"
#inputs, targets = generate_series(msi, 4)
X, Y = generate_series(stock, 8)
Y = Y.reshape((Y.shape[0],1))
ble=np.concatenate((X, Y), axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(ble)
X_scaled=scaled
size=X_scaled.shape[1]
Y=scaled[:,size-1]
X=scaled[:,[0,1,2,3,4,5,6,7,8]]
size=X.shape[1]
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True)
X_train = np.reshape(X_train, (-1, 1, size))
X_test = np.reshape(X_test, (-1, 1, size))
# =============================================================================

# NEURAL NETWORK
model = Sequential()
model.add(LSTM(64,input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adagrad')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

#training
history = model.fit(X_train, y_train, epochs=20, batch_size=5, validation_split=0.2, verbose=2)

#predictions for test data
predictions= model.predict(X_test)


# =============================================================================
#reshaping and inversing normalization
# =============================================================================
prediction_reshaped = np.zeros((len(predictions), size+1))
testY_reshaped = np.zeros((len(y_test), size+1))

prediction_r = np.reshape(predictions, (len(predictions),))
testY_r = np.reshape(y_test, (len(y_test),))

prediction_reshaped[:,size] = prediction_r
testY_reshaped[:,size] = testY_r

prediction_inversed = scaler.inverse_transform(prediction_reshaped)[:,size]
testY_inversed = scaler.inverse_transform(testY_reshaped)[:,size]
# =============================================================================

#calculating error rates
# =============================================================================
rmse = sqrt(mean_squared_error(testY_inversed, prediction_inversed))
maae=mean_absolute_error(testY_inversed, prediction_inversed)
r2=r2_score(testY_inversed,prediction_inversed) 
mape_err=mean(np.abs((testY_inversed - prediction_inversed) / testY_inversed)) * 100
# =============================================================================

model.output_shape 
model.summary()
model.get_config()
model.get_weights() 
#saving model
model.save('tesco_model.h5')

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(prediction_inversed[1:100], label='prognozy')
plt.plot(testY_inversed[1:100], label='wartości rzeczywiste')
plt.title('Prognozy w porównaniu z wartościami rzeczywistymi dla spółki SolarWinds', fontdict={'fontsize': 18, 'fontweight': 'medium'})
plt.xlabel('próbki 1-100', fontsize=15)
plt.ylabel('kurs akcji [USD]', fontsize=15)
plt.subplot(2,1,2)
plt.plot(prediction_inversed[1250:1350], label='prognozy')
plt.plot(testY_inversed[1250:1350], label='wartości rzeczywiste')
plt.title('  ', fontdict={'fontsize': 12, 'fontweight': 'medium'})
plt.xlabel('próbki 1250-1350', fontsize=15)
plt.ylabel('kurs akcji [USD]', fontsize=15)
plt.legend(loc='upper left', fontsize=13)
plt.show()
# =============================================================================

print('MAPE: %.3f' % mape_err)
print('Test RMSE: %.3f' % rmse)
print('Test MAE: %.3f' % maae)
print('R^2: %.3f' % r2)