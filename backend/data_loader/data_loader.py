import os
import json
import pandas as pd

def load_data():
    """Load and preprocess the data from JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "../init.json")

    with open(json_file_path, 'r') as infile:
        raw_data = json.load(infile)

    df = pd.DataFrame(raw_data if isinstance(raw_data, list) else [raw_data])
    df['ticker'] = df.get('ticker', 'NONE')
    df['ticker'] = df['ticker'].fillna('NONE')
    df['title'] = df['title'].fillna('')
    df['text'] = df['text'].fillna('')
    df["combined_text"] = df["title"] + " " + df["text"]
    
    return df

# we also have a data set, json file, with a bunch of company names and their respective ticker
company_map = {
    "tesla": "TSLA",
    "nio": "NIO",
    "apple": "AAPL",
    "gme": "GME",
    "amazon": "AMZN",
    "google": "GOOGL",
    "microsoft": "MSFT",
    "facebook": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "intel": "INTC",
    "uber": "UBER",
    "lyft": "LYFT",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "twitter": "TWTR",
    "snap": "SNAP",
    "pinterest": "PINS",
    "shopify": "SHOP",
    "spotify": "SPOT",
    "square": "SQ"
}

def map_company_to_ticker(user_query: str):
    words = user_query.lower().split()
    for company, ticker in company_map.items():
        if company in words:
            return ticker
    return None