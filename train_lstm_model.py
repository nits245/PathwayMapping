# train_lstm_model.py

# Section: Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from data_processing import DataProcessing

# Section: Load and prepare the dataset
scat_data = DataProcessing(filepath = 'datasets/Scats Data October 2006.xls')
X, y, scaler = scat_data.process_scats_data()

# Section: Split the dataset into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Section: Reshape input for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Section: Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=64, activation='tanh', input_shape=(X_train.shape[1], 1)))
lstm_model.add(Dense(1))

# Section: Compile the LSTM model
lstm_model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

# Section: Train the LSTM model
lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
)

# Predict using the test set
y_pred = lstm_model.predict(X_test)

# Section: Save the trained LSTM model
lstm_model.save('models/lstm_model.keras')

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")