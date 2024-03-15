import os
from dotenv import load_dotenv
import requests
import csv
from datetime import datetime, timedelta
import time

# Load environment variables from .env file
load_dotenv()

# Define your Aylien credentials
AppID = os.getenv('APP_ID')
APIKey = os.getenv('API_KEY')
PolygonAPIKey = os.getenv('POLYGON_API_KEY')

# Function to get authentication header
def get_auth_header(appid, apikey):
    return {
        'X-Application-Id': appid,
        'X-Application-Key': apikey
    }
# Function to fetch stories for specific companies within a date range
def fetch_stories_for_date_range(ticker, headers, start_date, end_date):
    all_stories = []

    params = {
        'entities.stock_tickers': ticker,
        'published_at.start': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'published_at.end': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'language': 'en',
        'per_page': 2,  # Set per_page to maximum allowed value
        'sort_by': 'published_at',
        'sort_direction': 'desc'
    }

    while True:
        time.sleep(1)  # Adding a 1-second delay between API calls
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
            print(f"Failed to fetch stories for ticker {ticker}: {response.status_code} - {response.text}")
            break

    return all_stories

# Function to fetch stock data for a given symbol within a date range using Polygon API
def get_stock_data(api_key, symbol, start_date, end_date):
    time.sleep(1)  # Adding a 1-second delay between API calls
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={api_key}"
    response = requests.get(base_url)
    if response.status_code == 200:
        data = response.json()
        results = data['results']
        stock_data = {datetime.fromtimestamp(result['t'] / 1000).strftime('%Y-%m-%d'): {'open': result['o'], 'close': result['c']} for result in results}
        return stock_data
    else:
        print(f"Failed to fetch stock data for {symbol} from Polygon API: {response.status_code} - {response.text}")
        return None

# Save data to CSV file
def save_data_to_csv(ticker, all_stories, stock_data):
    file_name = f"CurrentDatabase/{ticker}_db.csv"
    with open(file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow([
            'Publication Date','Summary', 'Sentiment Polarity', 'Sentiment Confidence', 'Keywords', 'stock_date', 'stock_price', 'percentage_change'
        ])

        for story in all_stories[:-1]:
            article_id = story.get('id', 'N/A')
            summary = ' '.join(story.get('summary', {}).get('sentences', []))
            keywords = ", ".join(story.get('keywords', []))

            sentiment = story.get('sentiment', {}).get('title', {})
            sentiment_polarity = sentiment.get('polarity', 'N/A')
            sentiment_confidence = story.get('sentiment', {}).get('body', {}).get('score', 'N/A')
            print(sentiment_polarity)
            publication_date = datetime.strptime(story.get('published_at', 'N/A'), '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
            stock_date = (datetime.strptime(publication_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

            
            if stock_data.get(stock_date) == None:
                stock_price = 'N/A'
                open_stock_price = 'N/A'
                print(stock_price)
            else:
                stock_price = stock_data.get(stock_date).get('close', 'N/A')
                open_stock_price = stock_data.get(stock_date).get('open', 'N/A')
                print(stock_price)

            if stock_price == 'N/A':
                # If current stock price is not available, try to get the previous day's price
                previous_date = (datetime.strptime(stock_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
                if stock_data.get(previous_date) == None:
                    stock_price = 'N/A'
                else:
                    previous_stock_price = stock_data.get(previous_date).get('close', 'N/A')
                    stock_price = previous_stock_price
                    open_stock_price = previous_stock_price
                #print("Current stock price not available. Previous day's price:", previous_stock_price)
                if stock_price == 'N/A':
                    # If current stock price is not available, try to get the previous day's price
                    previous_date = (datetime.strptime(stock_date, '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')

                    if stock_data.get(previous_date) == None:
                        stock_price = 'N/A'
                    else:
                        previous_stock_price = stock_data.get(previous_date).get('close', 'N/A')
                        stock_price = previous_stock_price
                        open_stock_price = previous_stock_price
                    #print("Current stock price not available. Previous day's price:", previous_stock_price)
                    if stock_price == 'N/A':
                        # If current stock price is not available, try to get the previous day's price
                        previous_date = (datetime.strptime(stock_date, '%Y-%m-%d') - timedelta(days=3)).strftime('%Y-%m-%d')
                        if stock_data.get(previous_date) == None:
                            stock_price = 'N/A'
                        else:
                            previous_stock_price = stock_data.get(previous_date).get('close', 'N/A')
                            stock_price = previous_stock_price
                            open_stock_price = previous_stock_price
                        
                        #print("Current stock price not available. Previous day's price:", previous_stock_price)
                    else:
                        None
                else:
                    None
            else:
                None
            percentage_change = ((float(stock_price) - float(open_stock_price)) / float(open_stock_price)) * 100
            
            writer.writerow([
                publication_date, summary, sentiment_polarity, sentiment_confidence, keywords, stock_date, stock_price, percentage_change
            ])

    print(f"Data has been written to {file_name}")

def main():
    tickers = ['AAPL','AMZN', 'TSLA', 'MSFT', 'AMD', 'GE', 'SMCI', 'META', 'BA','GOOGL','NVDA']  # Example tickers
    headers = get_auth_header(AppID, APIKey)
    # Define the date range (last 3 days)
    end_date = datetime.now()  # Current date
    #end_date = datetime.strptime(end_date, '%Y-%m-%d')
    start_date = datetime.now() - timedelta(days=7)  # 7 days ago

    # Fetch all stock data for each ticker in the date range
    for ticker in tickers:
        stock_data = get_stock_data(PolygonAPIKey, ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        print(stock_data)
        print(end_date)
        if stock_data:
            all_stories = []
            current_date = start_date
            while current_date < end_date:
                next_day = current_date + timedelta(days=1)
                if next_day > end_date:
                    next_day = end_date
                print(f"Fetching stories for {ticker}...")
                all_stories.extend(fetch_stories_for_date_range(ticker, headers, current_date, next_day))
                current_date = next_day
            save_data_to_csv(ticker, all_stories, stock_data)

if __name__ == '__main__':
    main()
