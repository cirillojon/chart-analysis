import yfinance as yf
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


def fetch_and_plot(ticker_symbol, start_date="2020-01-01", end_date="2023-01-01"):
    # Fetch data
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(start=start_date, end=end_date)
    # Plot data
    plt.figure(figsize=(12, 6))
    data['Close'].plot(title=f'{ticker_symbol} Stock Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (in currency)')
    plt.grid(True)
    plt.show()

# Example Usage
ticker = input("Enter the stock ticker: ")
fetch_and_plot(ticker)
