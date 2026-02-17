import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping

from data_processing import DataProcessing

# 1. Load & preprocess
dp = DataProcessing('datasets/Scats Data October 2006.xls')
X, y, scaler = dp.process_scats_data()

# 2. Split & reshape
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
# ensure shape = (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 3. Build GRU model
gru_model = Sequential([
    GRU(64, activation='tanh', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

gru_model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])

# 4. Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = gru_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)

# 5. Evaluate & save
y_pred = gru_model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"GRU MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
gru_model.save('models/gru_model.keras')
