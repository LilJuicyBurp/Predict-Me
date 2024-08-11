import pandas as pd
import os

def load_processed_data(ticker):
    try:
        data = pd.read_csv(f'processed_data/{ticker}_processed.csv', parse_dates=['Date'])
        print(f"Successfully loaded processed data for {ticker}")
        return data
    except Exception as e:
        print(f"Error loading processed data for {ticker}: {e}")
        raise

def load_sentiment_data(ticker):
    try:
        data = pd.read_csv(f'sentiment_data/{ticker}_sentiment.csv', parse_dates=['date'])
        print(f"Successfully loaded sentiment data for {ticker}")
        return data
    except Exception as e:
        print(f"Error loading sentiment data for {ticker}: {e}")
        raise

def combine_data(ticker):
    # Load processed stock data
    stock_data = load_processed_data(ticker)
    
    # Load sentiment data
    sentiment_data = load_sentiment_data(ticker)
    
    # Merge stock data and sentiment data on the date
    combined_df = stock_data.merge(sentiment_data, left_on='Date', right_on='date', how='left')
    
    # Identify the last month in the data
    last_month = combined_df['Date'].max().month
    last_year = combined_df['Date'].max().year
    
    # Set sentiment scores to NaN for all but the last month
    combined_df.loc[(combined_df['Date'].dt.month != last_month) | (combined_df['Date'].dt.year != last_year), 'sentiment'] = float('nan')
    
    # Fill NaN sentiment scores in the last month
    combined_df['sentiment'] = combined_df['sentiment'].fillna(method='ffill')
    
    # Drop the date column from sentiment data after merging
    combined_df.drop(columns=['date'], inplace=True)
    
    # Save the combined data
    combined_df.to_csv(f'combined_data/{ticker}_combined.csv', index=True)
    print(f'Combined data for {ticker} saved successfully.')

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "XOM", "WMT", "PG", "DIS"]
    os.makedirs('combined_data', exist_ok=True)
    
    for ticker in tickers:
        combine_data(ticker)
