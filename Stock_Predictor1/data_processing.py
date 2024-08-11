import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta

def add_technical_indicators(data):
    # Calculate moving averages
    data['MA10'] = data['Close'].rolling(window=10).mean().round(2)
    data['MA20'] = data['Close'].rolling(window=20).mean().round(2)
    data['MA30'] = data['Close'].rolling(window=30).mean().round(2)
    data['MA40'] = data['Close'].rolling(window=40).mean().round(2)
    data['MA50'] = data['Close'].rolling(window=50).mean().round(2)
    data['MA100'] = data['Close'].rolling(window=100).mean().round(2)
    data['MA200'] = data['Close'].rolling(window=200).mean().round(2)
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean().round(2)
    data['EMA25'] = data['Close'].ewm(span=25, adjust=False).mean().round(2)
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean().round(2)
    
    # Calculate RSI
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = (100 - (100 / (1 + rs))).round(2)
    
    # Calculate MACD
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = (ema12 - ema26).round(2)
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean().round(2)

    # Rounds Data
    data['Open'] = data['Open'].round(2)
    data['High'] = data['High'].round(2)
    data['Low'] = data['Low'].round(2)
    data['Close'] = data['Close'].round(2)
    data['Adj Close'] = data['Adj Close'].round(2)

    return data

def add_market_factors(data):
    # Download S&P 500 and VIX data
    sp500 = yf.download('^GSPC', start=data.index.min(), end=data.index.max())
    vix = yf.download('^VIX', start=data.index.min(), end=data.index.max())
    
    # Merge S&P 500 and VIX data with stock data
    data = data.merge(sp500['Close'].rename('SP500_Close').round(2), left_index=True, right_index=True, how='left')
    data = data.merge(vix['Close'].rename('VIX_Close').round(2), left_index=True, right_index=True, how='left')
    
    # Fill missing values
    data['SP500_Close'] = data['SP500_Close'].ffill()
    data['VIX_Close'] = data['VIX_Close'].ffill()
    
    return data

def process_data(ticker):
    try:
        # Load stock data
        data = pd.read_csv(f'data/{ticker}.csv', index_col=0, parse_dates=True)
        print(f"Successfully loaded data for {ticker}")
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Add market factors
        data = add_market_factors(data)
        
        # Drop the first 200 rows
        data = data.iloc[200:]
        
        # Fill any NaN values
        data = data.fillna(0)
        
        # Save the processed data
        data.to_csv(f'processed_data/{ticker}_processed.csv', index=True)
        print(f'Processed data for {ticker} saved successfully.')
    except Exception as e:
        print(f"Error processing data for {ticker}: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "XOM", "WMT", "PG", "DIS"]
    os.makedirs('processed_data', exist_ok=True)
    
    for ticker in tickers:
        process_data(ticker)
