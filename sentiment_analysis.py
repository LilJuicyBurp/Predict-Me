import requests
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
from datetime import datetime

nltk.download('vader_lexicon')

# Function to get news articles related to a ticker
def get_news(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey=fcd32084bf9a454686a9936c7f029ee9"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    return articles

# Function to analyze sentiment of news articles
def analyze_sentiment(articles):
    sia = SentimentIntensityAnalyzer()
    sentiment_data = []
    for article in articles:
        if article['description']:
            date = article['publishedAt'][:10]  # Extract the date part
            if datetime.strptime(date, "%Y-%m-%d") <= datetime(2024, 6, 30):  # Only consider articles from 2024
                score = sia.polarity_scores(article['description'])['compound']
                sentiment_data.append({'date': date, 'sentiment': score})
    return sentiment_data

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JNJ", "JPM", "XOM", "WMT", "PG", "DIS"]
    
    os.makedirs('sentiment_data', exist_ok=True)
    
    for ticker in tickers:
        articles = get_news(ticker)
        sentiment_data = analyze_sentiment(articles)
        # Save sentiment scores and dates to a CSV file
        pd.DataFrame(sentiment_data).to_csv(f'sentiment_data/{ticker}_sentiment.csv', index=False)
        print('Data Downlaoded for: ', ticker)