import requests
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

def get_stock_news(api_key, start_date, end_date, company):
    # Define the base URL for the API
    base_url = "http://eventregistry.org/api/v1/article/getArticles"

    # Define the parameters for the API request
    params = {
        "action": "getArticles",
        "keyword": f"{company} stock",
        "articlesCount": 100,
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "dataType": "news",
        "apiKey": api_key,
        "forceMaxDataTimeWindow": 180,
        "dateStart": start_date,
        "dateEnd": end_date
    }

    # Make the API request
    response = requests.get(base_url, params=params)
    print(response)
    
    # Parse the JSON response
    data = response.json()
    
    # Extract relevant information from the response
    articles = data['articles']['results']
    
    return articles

def get_article_details(api_key, article_uris):
    # Define the base URL for the API
    base_url = "http://eventregistry.org/api/v1/article/getArticle"

    # Initialize an empty list to store article details
    article_details = []

    # Loop through each article URI
    for article_uri in article_uris:
        # Define the parameters for the API request
        params = {
            "action": "getArticle",
            "articleUri": article_uri,
            "infoArticleBodyLen": -1,
            "resultType": "info",
            "apiKey": api_key
        }

        # Make the API request
        response = requests.post(base_url, json=params)

        # Parse the JSON response
        data = response.json()
        
        # Extract relevant information from the response
        if 'error' not in data:
            article_id = list(data.keys())[0]  # Get the article ID
            article_info = data[article_id]['info']  # Get the info dictionary

            # Extract title, date, and body from the info dictionary
            title = article_info.get('title', '')
            date = article_info.get('date', '')
            body = article_info.get('body', '')

            # Create a dictionary for the article details
            article_detail = {
                'title': title,
                'date': date,
                'body': body
            }

            # Append the article detail dictionary to the list
            article_details.append(article_detail)

    return article_details


def main():
    # Your NewsAPI.ai API key
    api_key = os.getenv("YOUR_API_KEY2")
    # Get user input for company name
    company = st.text_input('Company Name')
    
    # Get user input for date range
    start_date = st.date_input('Start Date')
    end_date = st.date_input('End Date')

    # Get stock market relevant news
    relevant_news = get_stock_news(api_key, start_date, end_date, company)

    # Display the selected articles using Streamlit
    st.title("Selected Stock Market News Articles")

    # Allow the user to select articles
    selected_articles = st.multiselect('Select Articles', [article['title'] for article in relevant_news])

    # Extract article URIs for selected articles
    selected_article_uris = [article['uri'] for article in relevant_news if article['title'] in selected_articles]

    # Get details of selected articles
    article_details = get_article_details(api_key, selected_article_uris)

    # Display selected article details
    for article in article_details:
        st.write(f"Title: {article['title']}")
        st.write(f"Date: {article['date']}")
        
        st.write(f"Body: {article['body']}")
        st.write('---')

if __name__ == "__main__":
    main()
