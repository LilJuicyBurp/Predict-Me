import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_combined_data(ticker):
    try:
        data = pd.read_csv(f'combined_data/{ticker}_combined.csv', parse_dates=['Date'], index_col='Date')
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        raise

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])
    return model

def train_model(ticker):
    # Load combined data
    data = load_combined_data(ticker)
    
    # Prepare features and target
    features = data[['MA10', 'MA20', 'MA30', 'MA40', 'MA50', 'EMA50', 'RSI', 'MACD', 'MACD_Signal', 'Volume', 'SP500_Close', 'VIX_Close']].values
    target = data['Close'].values
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Define sequence length
    seq_length = 60
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(features)):
        X.append(features[i-seq_length:i])
        y.append(target[i])
    
    X, y = np.array(X), np.array(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Save the trained model
    model.save(f'models/{ticker}_model.h5')
    print(f'Model for {ticker} saved successfully.')

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "XOM", "WMT", "PG", "DIS"]
    os.makedirs('models', exist_ok=True)
    
    for ticker in tickers:
        try:
            train_model(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
