# StockTalk

## Team Members
- Aayush Kumar 
- Abhishek Singh 
- Ghea Suyono

## Project Overview
“Do you want to make money? StockTalk is the right way to go.”

StockTalk is designed to streamline the investment process for individual investors by analyzing vast arrays of financial news and predicting stock price movements. It empowers users to make informed decisions with less stress and confusion.

## Technologies Used
- **Database Collection:**
  - News: Aylien News API
  - Stock Price: Polygon Stock API
- **Machine Learning Model:**
  - Sentiment Analysis: Utilized BERT (Bidirectional Encoder Representations from Transformers) through Hugging Face's transformers library.
  - Predictive Analysis: Applied BERT model to forecast stock price movements based on sentiment extracted from news articles.
- **Front-end:** Streamlit
- **Data Visualization:** Matplotlib and Altair for comparative analysis of actual vs. predicted stock price changes.

## How to Run
Open your terminal and execute the following commands:

```bash
# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # For Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

# Reflections
## What we learned:
- Understanding the intricate relationship between news and stock market performance.
- Mastering the art of news data analysis to craft predictive models.
- Grasping the significance of effective data visualization in simplifying investment decisions.

## Challenges faced:
- Accessing comprehensive and affordable news databases.
- Choosing a reliable stock price API for in-depth analysis.
- Overcoming technical hurdles in data integration and system performance.
- Balancing model complexity with intuitive user interface design.

StockTalk is not just a tool but a gateway to demystifying the stock market through intelligent analysis and predictive insights, empowering investors to navigate the financial world with confidence.
