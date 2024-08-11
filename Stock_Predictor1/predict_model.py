import pandas as pd
import numpy as np
import tensorflow as tf
import os

def load_combined_data(ticker):
    try:
        data = pd.read_csv(f'combined_data/{ticker}_combined.csv', index_col=0, parse_dates=True)
        print(f"Successfully loaded data for {ticker}")
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        raise

def load_model(ticker):
    try:
        model = tf.keras.models.load_model(f'models/{ticker}_model.h5')
        print(f"Successfully loaded model for {ticker}")
        return model
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        raise

def predict_and_print(ticker):
    # Load combined data
    data = load_combined_data(ticker)
    
    # Load trained model
    model = load_model(ticker)
    
    # Prepare features
    features = data[['MA50', 'MA200', 'Volume', 'sentiment']].values
    
    # Define sequence length
    seq_length = 60
    
    # Get the last sequence for prediction
    last_sequence = features[-seq_length:]
    
    # Predict for June 2024
    june_dates = pd.date_range(start='2024-06-03', end='2024-06-28', freq='B')
    future_predictions = []
    
    for _ in range(len(june_dates)):
        prediction = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
        future_predictions.append(prediction)
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, :-1] = features[-1, :-1]
        last_sequence[-1, -1] = prediction
    
    # Print predictions
    for date, price in zip(june_dates, future_predictions):
        print(f"Date: {date.date()}, Predicted Close Price: {price:.2f}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "XOM", "WMT", "PG", "DIS"]
    
    for ticker in tickers:
        try:
            predict_and_print(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
