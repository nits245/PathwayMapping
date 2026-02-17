# train_lstm_model.py
import keras.src.saving
from sklearn.model_selection import train_test_split
# Section: Import required libraries
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import losses
from data_processing import DataProcessing

import joblib

class Convolutional:
    def __init__(self, filepath, window_size=4):
        self.filepath = filepath
        self.window_size = window_size 
        self.dataproceser = DataProcessing(filepath)

    def data_process(self):
      
        X, y, scaler= self.dataproceser.process_scats_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        if X_train.ndim == 2: 
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        if X_test.ndim == 2: 
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        return X_train, X_test, y_train, y_test, scaler


    def train_model(self):
        X_train, X_test, y_train, y_test, scaler= self.data_process()

        num_features_for_conv1d = X_train.shape[2] 
        print(f"Shape of X_train: {X_train.shape}")
        print(f"Num features determined for Conv1D: {num_features_for_conv1d}") 

        simple_conv1d = Sequential([
            layers.Conv1D(filters=64, kernel_size=1, activation="relu", input_shape=(self.window_size, num_features_for_conv1d)),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(units=1, activation='relu'),
            layers.Dense(units=1)
        ])

        simple_conv1d.compile(optimizer='adam', loss=losses.Huber(), metrics=['mse'])

        simple_conv1d_history = simple_conv1d.fit(
            X_train, y_train,
            validation_split=0.10,
            epochs=50,
            batch_size=32
        )

        os.makedirs('models', exist_ok=True)
        simple_conv1d.save('models/conv1d.keras')


    def model_predict(self, new_data_sequence_raw, scaler_X, scaler_y, model_path='models/conv1d.keras'):
           raise NotImplementedError

if __name__ == '__main__':
    model = Convolutional('datasets/Scats Data October 2006.xls')
    model.train_model()
