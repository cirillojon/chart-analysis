import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Fetch stock data
def fetch_data(tickers, start_date="2020-01-01", end_date="2023-01-01"):
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return stock_data

# Calculate daily returns
def calculate_daily_returns(stock_data):
    return stock_data.pct_change().dropna()

# Objective function (Negative Sharpe Ratio)
def objective(weights, expected_returns, cov_matrix, rf_rate):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - rf_rate) / portfolio_std

# Portfolio optimization
def optimize_portfolio(daily_returns, rf_rate=0.01):
    num_assets = len(daily_returns.columns)
    args = (daily_returns.mean(), daily_returns.cov(), rf_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(objective, [1./num_assets for asset in range(num_assets)], args=args, constraints=constraints, bounds=bounds)
    return result

# Efficient Frontier
def plot_efficient_frontier(stock_data):
    daily_returns = calculate_daily_returns(stock_data)
    results = optimize_portfolio(daily_returns)
    
    portfolio_std = []
    portfolio_return = []
    
    # Monte Carlo simulation
    for _ in range(10000):
        weights = np.random.random(len(tickers))
        weights /= sum(weights)
        returns = np.dot(weights, daily_returns.mean())
        risk = np.sqrt(np.dot(weights.T, np.dot(daily_returns.cov(), weights)))
        portfolio_std.append(risk)
        portfolio_return.append(returns)

    plt.figure(figsize=(12, 6))
    plt.scatter(portfolio_std, portfolio_return, c=(np.array(portfolio_return)-0.01)/np.array(portfolio_std), marker='o', cmap='YlGnBu')
    plt.title('Efficient Frontier')
    plt.xlabel('Portfolio Risk')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

tickers = ["AAPL", "MSFT", "GOOGL"]
stock_data = fetch_data(tickers)
plot_efficient_frontier(stock_data)
