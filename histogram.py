import yfinance as yf
import matplotlib.pyplot as plt

def fetch_annual_returns(tickers, start_date="2020-01-01", end_date="2023-01-01"):
    annual_returns = []
    
    for ticker_symbol in tickers:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        # Calculate annual return
        years = (data.index[-1] - data.index[0]).days / 365.25
        total_return = data['Close'][-1] / data['Close'][0] - 1
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        annual_returns.append(annualized_return)
    
    return annual_returns

def plot_histogram(tickers):
    annual_returns = fetch_annual_returns(tickers)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, annual_returns, color=['blue', 'green', 'red'])
    plt.xlabel('Ticker')
    plt.ylabel('Annualized Return')
    plt.title('Annualized Returns of Given Tickers')
    plt.grid(axis='y')
    
    # Displaying the return percentages on top of the bars
    for i, v in enumerate(annual_returns):
        plt.text(i, v + 0.01, f"{v*100:.2f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.show()

# Example Usage
tickers = ["AAPL", "MSFT", "GOOGL"]  # Change these tickers as per your requirement
plot_histogram(tickers)
