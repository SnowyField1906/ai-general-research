import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Load data and preprocess
absolute_path = os.path.abspath(os.path.dirname(__file__))
data = pd.read_csv(absolute_path + '/bitcoin_test.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].apply(lambda x: time.mktime(x.timetuple()))

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

# Plot real data
real = scaler.inverse_transform(np.concatenate((sequences[:, -1, 0].reshape(-1, 1), targets.reshape(-1, 1)), axis=1))[:, 1]
plt.plot(real, color='red', label='Real')

# Plot LSTM model prediction
model = load_model(absolute_path + '/lstm_prediction.h5')
predicted_lstm = model.predict(sequences)
predicted_lstm = scaler.inverse_transform(np.concatenate((sequences[:, -1, 0].reshape(-1, 1), predicted_lstm), axis=1))[:, 1]
plt.plot(predicted_lstm, color='blue', label='Predicted (LSTM)')

# Predict with Fourier series least squares
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
