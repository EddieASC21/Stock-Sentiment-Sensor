import requests

# hard code mapping of ticker to company if the api expires
company_map = {
    "apple": "AAPL",
    "tesla": "TSLA",
    "nio": "NIO",
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

def get_ticker_from_api(company_name: str) -> str:
    api_key = "iAyH6422iBmfa0Fh6y2VQAuIP1SR6GkO"  
    url = f"https://financialmodelingprep.com/api/v3/search?query={company_name}&limit=1&exchange=NASDAQ&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        if results and isinstance(results, list) and len(results) > 0:
            ticker = results[0].get("symbol")
            return ticker
        else:
            return None
    except Exception as e:
        print("Error in get_ticker_from_api:", e)
        return None

def map_company_to_ticker(user_query: str) -> str:
    company_name = user_query.strip()
    ticker = get_ticker_from_api(company_name)
    if ticker is None:
        for comp, tkr in company_map.items():
            if comp.lower() in company_name.lower():
                return tkr
    return ticker
