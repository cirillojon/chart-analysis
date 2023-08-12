import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch stock data
ticker_symbol = "AAPL"
data = yf.download(ticker_symbol, start="2010-01-01", end="2023-01-01")['Adj Close']

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Create training dataset with 60 time-steps for each sample
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Split into training and test sets
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=3, batch_size=32)

# Predict using the test set
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Visualization
plt.figure(figsize=(14,6))
plt.plot(data.index[60:train_size], data[60:train_size], color='blue', label="Training data")
plt.plot(data.index[train_size+60:], data[train_size+60:], color='green', label="Actual Price")
plt.plot(data.index[train_size+60:], predicted_prices, color='red', label="Predicted Price")
plt.title(f"{ticker_symbol} Stock Price Prediction using LSTM")
plt.legend()
plt.show()
