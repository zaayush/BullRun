import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import csv
from dotenv import load_dotenv
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()
AppID = os.getenv('APP_ID')
APIKey = os.getenv('API_KEY')
PolygonAPIKey = os.getenv('POLYGON_API_KEY')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, tokenizer, max_length=512):
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx]
        text = f"{feature['Keywords']}"
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return inputs

def get_auth_header(app_id, api_key):
    return {
        'X-Application-Id': app_id,
        'X-Application-Key': api_key
    }

def fetch_stories_for_date_range(ticker, headers, start_date, end_date):
    all_stories = []
    params = {
        'entities.stock_tickers': ticker,
        'published_at.start': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'published_at.end': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'language': 'en',
        'per_page': 100,
        'sort_by': 'published_at',
        'sort_direction': 'desc'
    }

    while True:
        time.sleep(1)
        response = requests.get('https://api.aylien.com/news/stories', headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            stories = data.get('stories', [])
            if not stories:
                break
            all_stories.extend(stories)
            if 'next' in data.get('links', {}):
                params['cursor'] = data['links']['next']
            else:
                break
        else:
            break

    return all_stories

def get_stock_data(api_key, symbol, start_date, end_date):
    time.sleep(1)
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        return []

def predict_stock_price(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = {key: val.to(model.device) for key, val in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.extend(logits.cpu().numpy())
    return predictions

st.title("Stock News and Data Analysis")

tickers = ['AAPL', 'AMZN', 'TSLA', 'MSFT', 'AMD', 'GE', 'SMCI', 'META', 'BA']

tab1, tab2 = st.tabs(["News and Stock Data", "Predictive Stock Price"])

with tab1:
    selected_ticker = st.selectbox('Select a stock symbol:', tickers)
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=30))
    end_date = st.date_input("End date", datetime.now())

    if st.button('Fetch Data'):
        headers = get_auth_header(AppID, APIKey)
        stock_data = get_stock_data(PolygonAPIKey, selected_ticker, start_date, end_date)
        all_stories = fetch_stories_for_date_range(selected_ticker, headers, start_date, end_date)
        
        if stock_data and all_stories:
            st.write(f'Fetched {len(all_stories)} stories and stock data for {selected_ticker}.')
            st.dataframe(stock_data[:5])
            st.dataframe(all_stories[:5])

with tab2:
    st.write('Predictive stock price data will be displayed here.')
