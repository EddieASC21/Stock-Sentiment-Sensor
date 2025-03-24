import spacy

nlp_spacy = spacy.load("en_core_web_sm")

def custom_tokenizer(text):
    """Tokenize, lemmatize, and remove stopwords/punctuation using spaCy."""
    doc = nlp_spacy(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# my thought is that we have a data set with the words or the data set we have now, when cleaned and lemmatized is used as positive and negative sentiment 
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