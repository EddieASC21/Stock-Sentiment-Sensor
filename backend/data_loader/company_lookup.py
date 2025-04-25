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
company_map: dict[str,str] = {}

if isinstance(raw, dict):
    for key, entry in raw.items():
        title = entry["title"].strip().lower()
        ticker = entry["ticker"].strip().upper()
        company_map[title] = ticker

print(f"Loaded {len(company_map)} companies.")

# Build growing-prefix alias map
alias_map: dict[str,str] = {}

# Sort by number of words in the (raw) title
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

def map_company_to_ticker(user_query: str) -> str | None:
    q = re.sub(r"[^a-z0-9 ]", "", user_query.lower()).strip()
    if q in alias_map:
        return alias_map[q]
    # longest-prefix match
    for alias in sorted(alias_map, key=len, reverse=True):
        if q.startswith(alias):
            return alias_map[alias]
    return None

def valid_ticker(ticker: str) -> bool:
    return ticker in alias_map.values()

if __name__ == "__main__":
    # Smoke-test
    for name in ["apple", "apple inc", "apple hospitality", "pineapple", "pineapple express"]:
        print(f"{name!r} → {map_company_to_ticker(name)}")
