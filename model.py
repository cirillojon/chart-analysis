import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt


def fetch_and_plot(ticker_symbol, start_date="2020-01-01", end_date="2023-01-01", forecast_days=30):
    # Fetch data
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Prepare data for regression model
    days_since_start = (data.index - data.index[0]).days.values.reshape(-1, 1)
    
    # Train a linear regression model
    model = LinearRegression().fit(days_since_start, data['Close'])
    
    # Predict for the next `forecast_days` days
    future_days = np.array([days_since_start[-1] + i for i in range(1, forecast_days + 1)]).reshape(-1, 1)
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
    future_predictions = model.predict(future_days)
    
    # Plot data
    plt.figure(figsize=(12, 6))
    data['Close'].plot(label='Historical Data', title=f'{ticker_symbol} Stock Closing Prices & Prediction')
    plt.plot(future_dates, future_predictions, 'r', label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (in currency)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example Usage
ticker = input("Enter the stock ticker: ")
fetch_and_plot(ticker)
