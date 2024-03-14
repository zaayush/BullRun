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
import altair as alt
 
# Load environment variables from .env file
load_dotenv()
AppID = os.getenv('APP_ID')
APIKey = os.getenv('API_KEY')
PolygonAPIKey = os.getenv('POLYGON_API_KEY')
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

st.set_page_config(
    page_title="Stock News and Data Analysis",
    page_icon="📈",
    initial_sidebar_state="expanded",
)
 
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
 
# Main app interface
st.title("Stock News and Data Analysis")
tickers = ['AAPL', 'AMZN', 'TSLA', 'MSFT', 'AMD', 'BA', 'GOOGL', 'NVDA']
 
tab1, tab2 = st.tabs(["News and Stock Data", "Predictive Stock Price"])
 
with tab1:
    selected_ticker = st.selectbox('Select a stock symbol:', tickers)
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=30))
    end_date = st.date_input("End date", datetime.now())
 
    # Fetch Stock Data Button and functionality
    if st.button('Fetch Stock Data'):
        stock_data = get_stock_data(PolygonAPIKey, selected_ticker, start_date, end_date)
        if stock_data:
            stock_df = pd.DataFrame(stock_data)
            stock_df['date'] = pd.to_datetime(stock_df['t'], unit='ms').dt.date
            stock_df.rename(columns={'v': 'Volume', 'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low'}, inplace=True)
            st.subheader(f"Stock Data for {selected_ticker}")
            st.dataframe(stock_df.style.format(subset=['Open', 'Close', 'High', 'Low'], formatter="{:.2f}"))
           
            # Enhance the chart
            # Show chart title
            st.write(f"Stock Price Chart for {selected_ticker}")
            line_chart = alt.Chart(stock_df).mark_line().encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('Close:Q', title='Close Price'),
                tooltip=['date', 'Open', 'High', 'Low', 'Close', 'Volume']
            ).interactive().properties(
                width=800,
                height=400
            )
            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.error('Failed to fetch stock data. Please check the ticker or try again later.')
   
    # Initialize session state variables
    if 'story_index' not in st.session_state:
        st.session_state.story_index = 0  # Index to keep track of displayed stories
    if 'fetched_stories' not in st.session_state:
        st.session_state.fetched_stories = []
 
    with st.expander("News Stories", expanded=True):
        headers = get_auth_header(AppID, APIKey)
        # Fetch stories only if we haven't already, or if the "Fetch News Stories" button is pressed
        if st.button('Fetch News Stories') or not st.session_state.fetched_stories:
            st.session_state.fetched_stories = fetch_stories_for_date_range(selected_ticker, headers, start_date, end_date)
            st.session_state.story_index = 0  # Reset story index
    
        if st.session_state.fetched_stories:
            displayed_stories = st.session_state.fetched_stories[st.session_state.story_index:st.session_state.story_index + 5]
            for story in displayed_stories:
                st.markdown(f"**Title:** {story.get('title')}")
                st.markdown(f"**Summary:** {story.get('body')}")
                sentiment = story.get('sentiment', {}).get('polarity', 'neutral')
                sentiment_icon = "🔴" if sentiment == "negative" else "🟢" if sentiment == "positive" else "🟡"
                st.markdown(f"**Sentiment:** {sentiment_icon} {sentiment.capitalize()}")
                st.markdown(f"**Source:** {story.get('source', {}).get('name')}")
                st.markdown(f"**Published At:** {story.get('published_at')}")
                st.markdown("---")
    
            # Load More Stories Button
            if st.button('Load More Stories'):
                # Check if there are more stories to load
                if st.session_state.story_index + 5 < len(st.session_state.fetched_stories):
                    st.session_state.story_index += 5
                    st.rerun()
                else:
                    st.warning("No more stories to load.")
    
        else:
            st.error('No stories fetched. Please check the ticker or try a different date range.')
    
with tab2:
    stock_mapping = {
    "AAPL": {"csv_path": "CurrentDatabase/AAPL_db.csv", "model_path": "TrainedModels/saved_model_AAPL/"},
    "AMD": {"csv_path": "CurrentDatabase/AMD_db.csv", "model_path": "TrainedModels/saved_model_AMD/"},
    "GOOGL": {"csv_path": "CurrentDatabase/GOOGL_db.csv", "model_path": "TrainedModels/saved_model_GOOGL/"},
    "MSFT": {"csv_path": "CurrentDatabase/MSFT_db.csv", "model_path": "TrainedModels/saved_model_MSFT/"},
    "NVDA": {"csv_path": "CurrentDatabase/NVDA_db.csv", "model_path": "TrainedModels/saved_model_NVDA/"},
    "TSLA": {"csv_path": "CurrentDatabase/TSLA_db.csv", "model_path": "TrainedModels/saved_model_TSLA/"},
    "AMZN": {"csv_path": "CurrentDatabase/AMZN_db.csv", "model_path": "TrainedModels/saved_model_AMZN/"},
    "BA": {"csv_path": "CurrentDatabase/BA_db.csv", "model_path": "TrainedModels/saved_model_BA/"}
    
}

    # Select stock symbol from dropdown
    selected_stock = st.selectbox("Select a stock symbol:", list(stock_mapping.keys()))

    # Load the new data
    new_data = pd.read_csv(stock_mapping[selected_stock]["csv_path"])

    # Convert 'Sentiment Polarity' to numerical representation
    new_data['Sentiment Polarity'] = new_data['Sentiment Polarity'].map({'neutral': 0, 'positive': 1, 'negative': -1})

    # Convert 'Publication Date' and 'stock_date' to datetime objects
    new_data['Publication Date'] = pd.to_datetime(new_data['Publication Date'])
    new_data['stock_date'] = pd.to_datetime(new_data['stock_date'])

    # Use only required columns
    new_data = new_data[['Publication Date', 'Sentiment Polarity', 'Sentiment Confidence', 'Keywords', 'stock_date', 'percentage_change']]

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(stock_mapping[selected_stock]["model_path"])

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define custom dataset class
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, features, tokenizer, max_length=512):
            self.features = features
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            feature = self.features.iloc[idx]
            text = f"Publication Date: {feature['Publication Date']}, Sentiment Polarity: {feature['Sentiment Polarity']}, Sentiment Confidence: {feature['Sentiment Confidence']}, Keywords: {feature['Keywords']}, Stock Date: {feature['stock_date']}"
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                return_token_type_ids=False,
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return inputs

    # Create DataLoader for the new data
    new_dataset = CustomDataset(new_data, tokenizer)
    new_dataloader = DataLoader(new_dataset, batch_size=32)

    # Predictions
    predictions = []

    model.eval()
    with torch.no_grad():
        for batch_inputs in new_dataloader:
            batch_inputs = {key: val.squeeze(1).to(device) for key, val in batch_inputs.items()}
            outputs = model(**batch_inputs)
            logits = outputs.logits
            predictions.extend(logits.flatten().cpu().detach().numpy())

    # Convert predictions to percentage change
    predicted_percentage_change = predictions  # Modify this line as needed based on how your model is trained to predict percentage change

    # Get actual percentage change from the CSV file
    actual_percentage_change = new_data['percentage_change'].values

    # Predictions for tomorrow
    tomorrow_date = datetime.now() + timedelta(days=1)
    tomorrow_prediction = []

    with torch.no_grad():
        text = f"Publication Date: {tomorrow_date}, Sentiment Polarity: 0, Sentiment Confidence: 0, Keywords: None, Stock Date: None"
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        tomorrow_prediction = logits.item()

    import subprocess

    # Function to run cdb2.py script
    def run_cdb2_script():
        try:
            # Run the cdb2.py script using subprocess
            subprocess.run(["python", "CurrentDB.py"])
            st.write("Please wait a moment, updating current")
        except Exception as e:
            st.error(f"An error occurred while running the cdb2.py script: {e}")

    # Add a button to run the cdb2.py script
    if st.button("Fetch Latest Data"):
        run_cdb2_script()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot actual vs predicted percentage change
    ax.plot(new_data['stock_date'], actual_percentage_change, label='Actual Percentage Change', marker='o', linestyle='-')

    # Plot predicted percentage change if available
    if predicted_percentage_change:
        ax.plot(new_data['stock_date'], predicted_percentage_change, label='Predicted Percentage Change', marker='x', linestyle='--')

    # Plot tomorrow's prediction
    ax.plot(tomorrow_date, tomorrow_prediction, label='Tomorrow Prediction', marker='*', linestyle='--')

    # Draw a dotted green line from the last predicted percentage change to tomorrow's prediction if predictions are available
    if predicted_percentage_change:
        last_predicted_date = new_data['stock_date'].iloc[-1]
        last_predicted_change = predicted_percentage_change[-1]
        ax.plot([last_predicted_date, tomorrow_date], [last_predicted_change, tomorrow_prediction], 'g--')

    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Percentage Change')
    ax.set_title('Comparison of Actual vs Predicted Percentage Change')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)

    # Streamlit app
    st.title('Comparison of Actual vs Predicted Percentage Change')
    st.pyplot(fig)

