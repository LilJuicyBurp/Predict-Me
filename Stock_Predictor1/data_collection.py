import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Define the tickers and the period for the data
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "XOM", "WMT", "PG", "DIS"]
market_indices = ["^GSPC", "^VIX"]  # S&P 500 and VIX
end_date = datetime(2024, 7, 1)
start_date = end_date - timedelta(days=365*10)

# Create a directory to store the data
os.makedirs('data', exist_ok=True)

# Function to download and save data
def download_data(ticker):
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data.to_csv(f'data/{ticker}.csv')
    print(f'Downloaded data for {ticker}')

# Download and save the data for individual tickers
for ticker in tickers:
    download_data(ticker)

# Download and save the data for market indices
for index in market_indices:
    download_data(index)

