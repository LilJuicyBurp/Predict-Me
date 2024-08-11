import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta

def load_combined_data(ticker):
    try:
        data = pd.read_csv(f'combined_data/{ticker}_combined.csv', parse_dates=['Date'], index_col='Date')
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        raise

def predict_june(ticker):
    try:
        # Load combined data
        data = load_combined_data(ticker)
        
        # Prepare features
        features = data[['MA10', 'MA20', 'MA30', 'MA40', 'MA50', 'EMA50', 'RSI', 'MACD', 'MACD_Signal', 'Volume', 'SP500_Close', 'VIX_Close']]
        
        # Ensure all features are available
        if features.isnull().values.any():
            print(f"Missing data in features for {ticker}.")
            return
        
        # Feature scaling
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Define sequence length
        seq_length = 60
        
        # Create sequences for prediction
        X = []
        for i in range(seq_length, len(features)):
            X.append(features[i-seq_length:i])
        
        X = np.array(X)
        
        if len(X) == 0:
            print(f"Not enough data to create sequences for {ticker}")
            return
        
        # Load the trained model
        model = tf.keras.models.load_model(f'models/{ticker}_model.h5')
        
        # Make predictions
        predictions = model.predict(X)
        
        # Prepare a DataFrame to store predictions
        prediction_df = pd.DataFrame({
            'Date': data.index[seq_length:],
            'Prediction': predictions.flatten()
        })
        
        # Extract predictions for June 2024
        start_date = datetime(2024, 6, 1)
        end_date = datetime(2024, 6, 30)
        
        june_predictions = prediction_df[(prediction_df['Date'] >= start_date) & (prediction_df['Date'] <= end_date)]
        
        # Print the predictions
        print(f"Predictions for {ticker} for June 2024:")
        print(june_predictions)
    except Exception as e:
        print(f"Error predicting for {ticker}: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "XOM", "WMT", "PG", "DIS"]
    
    for ticker in tickers:
        predict_june(ticker)
