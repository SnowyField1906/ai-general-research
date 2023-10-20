import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load data and preprocess
absolute_path = os.path.abspath(os.path.dirname(__file__))
data = pd.read_csv(absolute_path + '/bitcoin_train.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].apply(lambda x: time.mktime(x.timetuple()))
# data = data[:int(len(data) * 0.4)]

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data[['Date', 'Close']])

# Prepare data for LSTM
sequence_length = 10 
sequences = []
targets = []
for i in range(len(data_normalized) - sequence_length):
    sequences.append(data_normalized[i:i+sequence_length])
    targets.append(data_normalized[i+sequence_length][1])

# Convert to numpy arrays
sequences = np.array(sequences)
targets = np.array(targets)

# Create LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 2)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(sequences, targets, epochs=100, batch_size=32, verbose=1)
# model.save(absolute_path + '/lstm-prediction.h5')

# Create Fourier series least squares model
num_terms = 2000
A = np.ones((len(sequences), num_terms * 2 + 1))
x_values = sequences[:, -1, 0]

for i in range(num_terms):
    A[:, 2 * i + 1] = np.sin(2 * np.pi * (i + 1) * x_values)
    A[:, 2 * i + 2] = np.cos(2 * np.pi * (i + 1) * x_values)

x = np.linalg.lstsq(A, targets, rcond=None)[0]
np.save(absolute_path + '/fourier_prediction.npy', x)

# Plot real data
x_axis = sequences[:, -1, 0].reshape(-1, 1)
real = scaler.inverse_transform(np.concatenate((x_axis, targets.reshape(-1, 1)), axis=1))[:, 1]
plt.plot(real, color='red', label='Real')

# Plot LSTM model prediction
model = load_model(absolute_path + '/lstm_prediction.h5')
predicted_lstm = model.predict(sequences)
predicted_lstm = scaler.inverse_transform(np.concatenate((sequences[:, -1, 0].reshape(-1, 1), predicted_lstm), axis=1))[:, 1]
plt.plot(predicted_lstm, color='blue', label='Predicted (LSTM)')

# Plot Fourier series least squares prediction
x = np.load(absolute_path + '/fourier_prediction.npy')
num_terms = 2000
x_values = sequences[:, -1, 0]
predicted_fourier = x[0]

for i in range(num_terms):
    predicted_fourier += x[2 * i + 1] * np.sin(2 * np.pi * (i + 1) * x_values)
    predicted_fourier += x[2 * i + 2] * np.cos(2 * np.pi * (i + 1) * x_values)

predicted_fourier = scaler.inverse_transform(np.concatenate((sequences[:, -1, 0].reshape(-1, 1), predicted_fourier.reshape(-1, 1)), axis=1))[:, 1]
plt.plot(predicted_fourier, color='green', label='Predicted (Fourier)')

plt.legend()
plt.show()