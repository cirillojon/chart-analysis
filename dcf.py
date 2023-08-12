import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Constants and assumptions
discount_rate = 0.1  # This would be replaced by WACC later
terminal_growth_rate = 0.03
forecast_period = 5

# Fetch financial data
ticker_symbol = input("Enter the stock ticker: ")
ticker = yf.Ticker(ticker_symbol)
financials = ticker.financials.transpose()

# Print available columns to debug
print("Available columns in financials:")
print(financials.columns)

# Use Operating Income as a proxy for Free Cash Flow
if "Operating Income" in financials.columns:
    recent_free_cash_flow = financials["Operating Income"].dropna()[0]
else:
    raise ValueError("Couldn't fetch or compute the required financial data.")

yearly_growth = financials["Operating Income"].pct_change().mean()
free_cash_flows = [recent_free_cash_flow * (1 + yearly_growth)**i for i in range(forecast_period)]

# Terminal cash flow
terminal_cash_flow = free_cash_flows[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
present_values = [cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(free_cash_flows)]
terminal_value = terminal_cash_flow / (1 + discount_rate) ** forecast_period
total_present_value = np.sum(present_values) + terminal_value

# Metrics computations
if "Normalized EBITDA" in financials.columns:
    ebitda = financials["Normalized EBITDA"].dropna()[0]
else:
    ebitda = None  

ebit = recent_free_cash_flow
tax_rate = 0.3
ebiat = ebit * (1 - tax_rate)

years_available = len(financials["Operating Income"].dropna())
if years_available > 1:
    start_value = financials["Operating Income"].dropna().iloc[-1]
    end_value = ebit
    cagr = ((end_value/start_value) ** (1/years_available)) - 1
else:
    cagr = yearly_growth

wacc = 0.1

print(f"CAGR: {cagr}")
print(f"EBITDA: {ebitda}")
print(f"EBIT: {ebit}")
print(f"EBIAT: {ebiat}")
print(f"WACC: {wacc}")

# Visualization
plt.figure(figsize=(12, 6))
plt.bar(range(len(free_cash_flows)), free_cash_flows, alpha=0.6, label="Projected FCF", color='blue')
plt.bar(range(len(present_values)), present_values, alpha=0.6, label="Discounted FCF", color='green')
plt.axhline(y=terminal_cash_flow, color='r', linestyle='-', label="Terminal Cash Flow")
plt.title(f"DCF Analysis for {ticker_symbol}")
plt.xlabel("Year")
plt.ylabel("Value (in currency units)")
plt.xticks(range(len(free_cash_flows)), [f"Y{i+1}" for i in range(len(free_cash_flows))])
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print(f"Total Present Value: {total_present_value}")
