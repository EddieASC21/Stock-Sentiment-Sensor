import json
import re
from pathlib import Path
from typing import Optional

BASE = Path(__file__).parent
FILE_PATH = BASE / "company_and_tickers_map.json"

if not FILE_PATH.exists():
    raise FileNotFoundError(f"Cannot find {FILE_PATH!r}")

with FILE_PATH.open("r", encoding="utf-8") as f:
    raw = json.load(f)

company_map: dict[str,str] = {}

if isinstance(raw, dict):
    for key, entry in raw.items():
        title = entry["title"].strip().lower()
        ticker = entry["ticker"].strip().upper()
        company_map[title] = ticker

print(f"Loaded {len(company_map)} companies.")

alias_map: dict[str,str] = {}

sorted_companies = sorted(
    company_map.items(),
    key=lambda kv: len(kv[0].split())
)

for full_name, ticker in sorted_companies:
    # strip out punctuation and split into words
    clean = re.sub(r"[^a-z0-9 ]", "", full_name)
    words = clean.split()
    for i in range(1, len(words) + 1):
        alias = " ".join(words[:i])
        # only set on first encounter
        alias_map.setdefault(alias, ticker)

def map_company_to_ticker(user_query: str) -> Optional[str]:
    q = re.sub(r"[^a-z0-9 ]", "", user_query.lower()).strip()
    if q in alias_map:
        return alias_map[q]
    # any alias anywhere
    for alias, ticker in alias_map.items():
        if alias in q:
            return ticker
    return None


def valid_ticker(ticker: str) -> Optional[bool]:
    return ticker in alias_map.values()

if __name__ == "__main__":
    # Smoke-test
    print(f"'apple' → {map_company_to_ticker('apple')}")
    print(f"'apple inc' → {map_company_to_ticker('apple inc')}")
    print(f"'apple hospitality' → {map_company_to_ticker('apple hospitality')}")
    print(f"'pineapple' → {map_company_to_ticker('pineapple')}") # Should map to Pinapple Inc.
    print(f"'pineapple express' → {map_company_to_ticker('pineapple express')}") # Should map to Pineapple Express Cannabis
    print(f"'who' → {map_company_to_ticker('who')}") # Should now likely be None
    print(f"'western' → {map_company_to_ticker('western')}") # Example that might have been problematic before
    print(f"'western petroleum' → {map_company_to_ticker('western petroleum')}")
