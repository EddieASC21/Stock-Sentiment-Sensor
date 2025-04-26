import spacy
import pandas as pd
import re
from data_loader import load_data

nlp_spacy = spacy.load("en_core_web_sm")

def custom_tokenizer(text):
    doc = nlp_spacy(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

extended_positive = {
    "bullish": ["bullish", "optimistic", "confident", "upbeat", "growth", "gain", "soar", "strong", "improving", "resilient"],
    "buy":     ["buy", "purchase", "accumulate", "long", "invest", "hold"],
    "moon":    ["moon", "rocket", "rally", "surge", "explode", "skyrocket"],
    "call":    ["call", "calls", "upgrade", "outperform", "positive", "improve"]
}

extended_negative = {
    "bearish": ["bearish", "pessimistic", "uncertain", "fear", "down", "decline", "drop", "weak", "struggling"],
    "sell":    ["sell", "dump", "liquidate", "short", "divest"],
    "bad":     ["bad", "weak", "poor", "loss", "fall", "slump"],
    "put":     ["put", "puts", "downgrade", "underperform", "negative"]
}

def flatten_syns(syn_dict):
    out = set()
    for main_word, synonyms in syn_dict.items():
        out.add(main_word)
        out.update(synonyms)
    return list(out)

pos_keywords = flatten_syns(extended_positive)
neg_keywords = flatten_syns(extended_negative)

def weak_label(text):
    lower_text = text.lower()
    if any(word in lower_text for word in pos_keywords):
        return 1
    elif any(word in lower_text for word in neg_keywords):
        return 0
    else:
        return None

def highlight_top_words(text, top_words):
    tokens = text.split()
    new_tokens = []
    for t in tokens:
        t_clean = t.lower().strip(".,!?;:")
        if t_clean in top_words:
            new_tokens.append(f"<b>{t}</b>")
        else:
            new_tokens.append(t)
    return " ".join(new_tokens)

def explain_post_sentiment(post_text):
    tokens = [t.lower().strip(".,!?;:") for t in post_text.split()]
    pos_found = {t for t in tokens if t in pos_keywords}
    neg_found = {t for t in tokens if t in neg_keywords}
    explanation_parts = []
    if pos_found:
        explanation_parts.append("Positive keywords: " + ", ".join(pos_found))
    if neg_found:
        explanation_parts.append("Negative keywords: " + ", ".join(neg_found))
    if not explanation_parts:
        return "No significant sentiment keywords found."
    return " | ".join(explanation_parts)

# Example stock/company keywords
STOCK_KEYWORDS = [
    
    # General finance terms
    "share price", "dividends",
    "capital gains", "earnings", "EPS", "PE ratio", "P/E ratio", "valuation",
    "undervalued", "overvalued", "buyback", "stock split",
    "portfolio", "revenue", "profit", "gross margin", 
    "cash flow", "balance sheet", "assets", "liabilities", "debt", "liquidity",
    
    # Trading terms
    "bullish", "bearish", "short squeeze", "pump and dump", "volatility",
    "support level", "resistance level", "moving average", "RSI", "MACD", "trendline", "breakout",
    "downtrend", "uptrend", "sector rotation", "technical analysis", "fundamental analysis",
    
    # Investment actions
    "buy", "sell", "holding", "covered call", "options",
    "calls", "puts", "leaps", "hedge", "diversify", "risk management", "stop loss",
    "target price", "price target", "analyst rating", "upgrade", "downgrade",
    
    # Financial events
    "earnings report", "quarterly results", "guidance", "outlook", "filing", "10-K", "10-Q",
    "dividend", "stock offering", "bankruptcy", "restructuring",
    
    # Economic indicators (optional but could be useful)
    "inflation", "interest rates", "GDP", "unemployment", "consumer spending", "CPI", "FOMC", "Federal Reserve"
]

SLANG_WORDS = [
    "hodl", "lmao", "lol", "moon", "to the moon", "rocket", "🚀",
    "bagholder", "bagholders", "paper hands", "diamond hands", "tendies",
    "stonks", "yolo", "YOLO", "fomo", "FOMO", "rekt", "pump it", "pump",
    "dump", "pump and dump", "buy the dip", "BTD", "rip", "moonshot",
    "diamond hands only", "paperhands", "hold the line", "apes together strong",
    "ape gang", "smash buy", "send it", "going parabolic", "moonboy", "moonboys",
    "penny stock hero", "lambo", "when lambo", "shill", "shilling", "no-brainer",
    "it's over", "we're so back", "bag secure", "bag secured", "no way it drops",
    "no way it falls", "hold hold hold", "load the boat", "back up the truck",
    "printing money", "easy money", "stonk gods", "stonk lords",
    "just vibes", "straight vibin", "gaslighting"
]



# Precompile a single regex pattern
stock_keywords_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in STOCK_KEYWORDS) + r')\b', flags=re.IGNORECASE)
slang_words_pattern = re.compile(r'(?:' + '|'.join(re.escape(word) for word in SLANG_WORDS) + r')', flags=re.IGNORECASE)

def clean_comments(df):
    print("Filtering comments...")

    # Only keep strings
    df = df[df['combined_text'].apply(lambda x: isinstance(x, str))]
    print(f"Remaining comments after type check: {len(df)}")
    
    # Remove short comments based on word count
    df = df[df['combined_text'].apply(lambda x: len(x.split()) >= 20)]
    print(f"Remaining comments after word count check: {len(df)}")
    
    # Check stock keywords using the compiled regex
    df = df[df['combined_text'].str.contains(stock_keywords_pattern, na=False)]
    print(f"Remaining comments after stock keyword check: {len(df)}")
    
    # Remove slang words using the compiled regex
    df = df[~df['combined_text'].str.contains(slang_words_pattern, na=False)]
    print(f"Remaining comments after slang word check: {len(df)}")
    
    # Remove duplicates
    df_filtered = df.drop_duplicates(subset=['combined_text']).copy()

    df_filtered.reset_index(drop=True, inplace=True)
    print(f"Remaining comments after deduplication: {len(df_filtered)}")
    return df_filtered
