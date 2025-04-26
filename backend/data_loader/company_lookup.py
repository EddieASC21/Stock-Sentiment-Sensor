import json
import re
from pathlib import Path

BASE = Path(__file__).parent
FILE_PATH = BASE / "company_and_tickers_map.json"

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Cannot find {FILE_PATH!r}")

# Load
with FILE_PATH.open("r", encoding="utf-8") as f:
    raw = json.load(f)

# Normalize into simple { title: ticker } dict
company_map: dict[str, str] = {}

if isinstance(raw, dict):
    for key, entry in raw.items():
        title = entry["title"].strip().lower()
        ticker = entry["ticker"].strip().upper()
        company_map[title] = ticker

print(f"Loaded {len(company_map)} companies.")

# Build word-based prefix alias map
alias_map: dict[str, str] = {}

# Sort by number of words in the (raw) title
sorted_companies = sorted(
    company_map.items(),
    key=lambda kv: len(kv[0].split())
)

for full_name, ticker in sorted_companies:
    clean = re.sub(r"[^a-z0-9 ]", "", full_name)
    words = clean.split()
    for i in range(1, len(words) + 1):
        alias = " ".join(words[:i])
        alias_map.setdefault(alias, ticker)

def map_company_to_ticker(user_query: str) -> str | None:
    q = re.sub(r"[^a-z0-9 ]", "", user_query.lower()).strip()
    if q in alias_map:
        return alias_map[q]
    # Word-based prefix match
    query_words = q.split()
    for i in range(len(query_words), 0, -1):
        prefix = " ".join(query_words[:i])
        if prefix in alias_map:
            return alias_map[prefix]
    return None

def valid_ticker(ticker: str) -> bool:
    if ticker is None:
        return False
    if ticker == "":
        return False
    if ticker == "NONE":
        return False
    if ticker == "N/A":
        return False
    return ticker in alias_map.values()

if __name__ == "__main__":
    # Smoke-test
    print(f"'apple' → {map_company_to_ticker('apple')}")
    print(f"'apple inc' → {map_company_to_ticker('apple inc')}")
    print(f"'apple hospitality' → {map_company_to_ticker('apple hospitality')}")
    print(f"'pineapple' → {map_company_to_ticker('pineapple')}")
    print(f"'pineapple express' → {map_company_to_ticker('pineapple express')}")
    print(f"'who' → {map_company_to_ticker('who')}") # Should now likely be None
    print(f"'western' → {map_company_to_ticker('western')}") # Example that might have been problematic before
    print(f"'western petroleum' → {map_company_to_ticker('western petroleum')}")