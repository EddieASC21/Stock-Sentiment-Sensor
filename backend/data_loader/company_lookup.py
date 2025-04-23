import requests
import yfinance as yf

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
    "square": "SQ",
    "berkshire hathaway": "BRK.A",
    "jpmorgan chase": "JPM",
    "walmart": "WMT",
    "coca‑cola": "KO",
    "pepsi": "PEP",
    "disney": "DIS",
    "nike": "NKE",
    "starbucks": "SBUX",
    "paypal": "PYPL",
    "visa": "V",
    "mastercard": "MA",
    "procter & gamble": "PG",
    "johnson & johnson": "JNJ",
    "ibm": "IBM",
    "exxon mobil": "XOM",
    "chevron": "CVX",
    "pfizer": "PFE",
    "merck": "MRK",
    "google parent": "GOOG",
    "alphabet": "GOOGL",
    "at&t": "T",
    "verizon": "VZ",
    "comcast": "CMCSA",
    "charter communications": "CHTR",
    "netflix": "NFLX",
    "pepsico": "PEP",
    "caterpillar": "CAT",
    "3m": "MMM",
    "dow": "DOW",
    "boeing": "BA",
    "general electric": "GE",
    "gm": "GM",
    "ford": "F",
    "tesla motors": "TSLA",
    "elon musk": "TSLA",
    "zoom": "ZM",
    "square": "SQ",
    "robinhood": "HOOD",
    "coinbase": "COIN",
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

def get_ticker_via_yfinance(name: str) -> str | None:
    try:
        searcher = yf.Search(name)
        if searcher.results:
            return searcher.results[0]['symbol']
        
        candidate = yf.Ticker(name)
        hist = candidate.history(period="1d")
        if not hist.empty:
            return name.upper()
    except Exception:
        pass
    return None
    
def map_company_to_ticker(user_query: str) -> str | None:
    """Map a company name or search query to a ticker symbol"""
    name = user_query.strip()

    for comp, tkr in company_map.items():
        if comp.lower() in name.lower():
            return tkr
    
    ticker = get_ticker_via_yfinance(name)
    if ticker:
        return ticker

    ticker = get_ticker_from_api(name)
    if ticker:
        return ticker

    return None
