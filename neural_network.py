import tkinter as tk
from tkinter import simpledialog, messagebox
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras import models
from keras import layers
from datetime import datetime, timedelta

def fetch_and_predict(ticker_symbol):
    # Fetch stock data
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
    model = models.Sequential()
    model.add(layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(layers.LSTM(units=50))
    model.add(layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=3, batch_size=32)

    # Predict using the test set
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Predict the next year's prices
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(252)]
    future_predictions = []
    input_sequence = scaled_data[-60:].tolist()
    
    for _ in range(252):
        next_prediction = model.predict(np.array(input_sequence[-60:]).reshape(1, 60, 1))
        input_sequence.append(next_prediction[0])
        future_predictions.append(next_prediction[0])

    future_predictions = scaler.inverse_transform(future_predictions)

    return data[train_size+60:], predicted_prices, future_dates, future_predictions, data.index[60:train_size], data[60:train_size]

def show_plot(train_indexes, training_data, actual, predicted, future_dates, future_predictions):
    plt.figure(figsize=(14, 6))
    plt.plot(train_indexes, training_data, color='blue', label="Training data")
    plt.plot(actual.index, actual.values, color='green', label="Actual Price")
    plt.plot(actual.index, predicted, color='red', label="Predicted Price")
    plt.plot(future_dates, future_predictions, color='purple', label="Future Predictions", linestyle='dashed')
    plt.legend()
    plt.show()

def on_submit():
    ticker_symbol = ticker_input.get()
    if not ticker_symbol:
        messagebox.showerror("Error", "Please enter a valid ticker symbol.")
        return

    try:
        actual, predicted, future_dates, future_predictions, train_indexes, training_data = fetch_and_predict(ticker_symbol)
        show_plot(train_indexes, training_data, actual, predicted, future_dates, future_predictions)
    except Exception as e:
        messagebox.showerror("Error", str(e))

app = tk.Tk()
app.title("Stock Price Prediction")

frame = tk.Frame(app)
frame.pack(padx=10, pady=10)

tk.Label(frame, text="Enter Stock Ticker:").grid(row=0, column=0, padx=5, pady=5)
ticker_input = tk.Entry(frame)
ticker_input.grid(row=0, column=1, padx=5, pady=5)

submit_btn = tk.Button(frame, text="Predict & Plot", command=on_submit)
submit_btn.grid(row=1, columnspan=2, pady=10)

app.mainloop()
