import streamlit as st
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
load_dotenv()

def get_stock_data(api_key, symbol):
    base_url = "https://www.alphavantage.co/query"
    date_six_months_ago = datetime.now() - timedelta(days=50)
    date_six_months_ago_str = date_six_months_ago.strftime("%Y-%m-%d")
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    stock_data = data['Time Series (Daily)']
    filtered_data = {date: stock_data[date] for date in stock_data if date >= date_six_months_ago_str}
    return filtered_data

def plot_stock_data(symbol, data):
    dates = list(data.keys())
    open_prices = [float(data[date]['1. open']) for date in dates]
    high_prices = [float(data[date]['2. high']) for date in dates]
    low_prices = [float(data[date]['3. low']) for date in dates]
    close_prices = [float(data[date]['4. close']) for date in dates]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(dates[::-1], open_prices[::-1], label='Open')
    ax.plot(dates[::-1], high_prices[::-1], label='High')
    ax.plot(dates[::-1], low_prices[::-1], label='Low')
    ax.plot(dates[::-1], close_prices[::-1], label='Close')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'Stock Prices for {symbol}')
    ax.set_xticklabels(dates[::-1], rotation=45)
    ax.legend()
    st.pyplot(fig)

def main():
    api_key = os.getenv("YOUR_API_KEY1")
    top_20_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "BRK.A", "JPM", "JNJ",
                     "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "INTC", "CMCSA"]

    selected_stock = st.selectbox("Select a stock", top_20_stocks)
    st.write(f"Fetching data for {selected_stock}...")
    stock_data = get_stock_data(api_key, selected_stock)
    plot_stock_data(selected_stock, stock_data)

if __name__ == "__main__":
    main()
