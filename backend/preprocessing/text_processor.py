import spacy
import pandas as pd
import re
from collections import Counter

nlp_spacy = spacy.load("en_core_web_sm")

def custom_tokenizer(text):
    doc = nlp_spacy(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

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

def explain_post_sentiment(post_text, clf_pipeline):
    try:
        # Use the pipeline to predict
        prediction = clf_pipeline.predict([post_text])[0]
        
        # Get probability scores if available
        try:
            probas = clf_pipeline.predict_proba([post_text])[0]
            confidence = max(probas) * 100
            confidence_str = f" (confidence: {confidence:.1f}%)"
        except:
            confidence_str = ""
        
        # Get TF-IDF vectorizer from pipeline
        vectorizer = clf_pipeline.named_steps['tfidf']
        tfidf_vector = vectorizer.transform([post_text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_vector.toarray()[0]
        
        # Sort indices by TF-IDF score (highest first)
        sorted_indices = tfidf_scores.argsort()[::-1]
        
        # Get top 5 terms with non-zero weights
        key_terms = []
        for idx in sorted_indices:
            if tfidf_scores[idx] > 0 and len(key_terms) < 5:
                key_terms.append(feature_names[idx])
            if len(key_terms) >= 5:
                break
        
        # Build explanation
        explanation = f"Sentiment: {prediction}{confidence_str}\n"
        if key_terms:
            explanation += "Key terms: " + ", ".join(key_terms)
        else:
            explanation += "No key terms identified."
        
        return explanation
    
    except Exception as e:
        return "No explanation available due to error"

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
