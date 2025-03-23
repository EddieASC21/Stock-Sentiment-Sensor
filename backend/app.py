import os
import json
import pandas as pd
import numpy as np
import re

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

app = Flask(__name__)
CORS(app)

nlp_spacy = spacy.load("en_core_web_sm")

def custom_tokenizer(text):
    # we are now using spaCy to tokenize, lemmatize, and remove stopwords/punctuation instead of sparks nlp
    doc = nlp_spacy(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return tokens

current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, "init.json")

with open(json_file_path, 'r') as infile:
    raw_data = json.load(infile)

df = pd.DataFrame(raw_data if isinstance(raw_data, list) else [raw_data])
df['ticker'] = df.get('ticker', 'NONE')
df['ticker'] = df['ticker'].fillna('NONE')
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

df["combined_text"] = df["title"] + " " + df["text"]

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

df["label"] = df["combined_text"].apply(weak_label)
df = df.dropna(subset=["label"])

# we are now logistical regression and TF-IDF instead of Naives Bayes for our classification
clf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english')),
    ('clf', LogisticRegression())
])
clf_pipeline.fit(df["combined_text"], df["label"])

# we are now ranking with the help of TF-IDF and SVD instead of PCA
rank_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english')),
    ('svd', TruncatedSVD(n_components=50))
])
rank_vectors = rank_pipeline.fit_transform(df["combined_text"])
df["rank_vector"] = list(rank_vectors)

def compute_cosine(v1, v2):
    return sk_cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]

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

def analyze_sentiment_for_ticker(ticker: str) -> str:
    sub_df = df[df['ticker'].str.upper() == ticker.upper()]
    if sub_df.empty:
        return f"<p>No recent discussion found for ticker: <b>{ticker.upper()}</b>.</p>"
    
    predictions = clf_pipeline.predict(sub_df["combined_text"])
    sub_df = sub_df.copy()
    sub_df["prediction"] = predictions

    pos_count = (sub_df["prediction"] == 1).sum()
    neg_count = (sub_df["prediction"] == 0).sum()
    total = pos_count + neg_count
    if total == 0:
        return f"<p>Sentiment about <b>{ticker.upper()}</b> is unclear or neutral.</p>"
    tone = "Positive" if pos_count > neg_count else "Negative" if neg_count > pos_count else "Mixed"

    pos_word_count = {}
    neg_word_count = {}
    for text in sub_df["combined_text"]:
        for token in text.lower().split():
            token_clean = token.strip(".,!?;:")
            if token_clean in pos_keywords:
                pos_word_count[token_clean] = pos_word_count.get(token_clean, 0) + 1
            if token_clean in neg_keywords:
                neg_word_count[token_clean] = neg_word_count.get(token_clean, 0) + 1

    sorted_pos = sorted(pos_word_count.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_neg = sorted(neg_word_count.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def build_keywords_html(sorted_list):
        if not sorted_list:
            return "<p>None</p>"
        max_count = sorted_list[0][1]
        html = "<ul>"
        for i, (word, count) in enumerate(sorted_list, start=1):
            weight = round(count / max_count, 2)
            html += f"<li>{i}. <b>{word}</b> ({count}) - weight: {weight}</li>"
        html += "</ul>"
        return html

    pos_html = build_keywords_html(sorted_pos)
    neg_html = build_keywords_html(sorted_neg)
    
    pos_top_set = {word for word, _ in sorted_pos}
    neg_top_set = {word for word, _ in sorted_neg}
    all_top_set = pos_top_set.union(neg_top_set)
    
    posts_html = "<ol>"
    top_posts = sub_df.head(5)
    for idx, row in top_posts.iterrows():
        classification = "Positive" if row["prediction"] == 1 else "Negative"
        full_text = row["combined_text"].strip()
        highlighted_text = highlight_top_words(full_text, all_top_set)
        explanation = explain_post_sentiment(full_text)
        # If a URL is available in the row, add it as a hyperlink.
        url_link = row.get("url", "")
        url_html = f"<br><a href='{url_link}' target='_blank'>Reddit Post</a>" if url_link and isinstance(url_link, str) and url_link.strip() != "" else ""
        posts_html += f"<li><strong>{classification}</strong> - {highlighted_text}{url_html}<br><em>{explanation}</em></li>"
    posts_html += "</ol>"

    output_html = f"""
    <div class="sentiment-container">
      <h2>{ticker.upper()} Sentiment Analysis</h2>
      <p><strong>Total relevant posts:</strong> {total}</p>
      <p><strong>Overall Sentiment:</strong> {tone}</p>
      <p><strong>Positive Mentions:</strong> {pos_count}</p>
      <p><strong>Negative Mentions:</strong> {neg_count}</p>

      <h3>Top 10 Positive Keywords:</h3>
      {pos_html}

      <h3>Top 10 Negative Keywords:</h3>
      {neg_html}

      <h3>Top 5 Relevant Posts:</h3>
      {posts_html}

      <p><em>Based on our model and keyword analysis, the community's opinion is {tone.lower()} on {ticker.upper()}.</em></p>
    </div>
    """
    return output_html

def search_documents(query: str, top_n: int = 10):
    query_vec = rank_pipeline.transform([query])[0]
    similarities = []
    for idx, row in df.iterrows():
        doc_vec = row["rank_vector"]
        sim = compute_cosine(np.array(query_vec), np.array(doc_vec))
        similarities.append(sim)
    df["sim_score"] = similarities
    results = df.sort_values("sim_score", ascending=False).head(top_n)
    results_html = "<ol>"
    for _, row in results.iterrows():
        results_html += f"<li><strong>{row['ticker']}</strong> - {row['title']}<br><em>Score: {round(row['sim_score'],2)}</em></li>"
    results_html += "</ol>"
    return results_html

def rank_keywords_for_ticker(ticker: str, top_n: int = 10):
    sub_df = df[df['ticker'].str.upper() == ticker.upper()]
    if sub_df.empty:
        return f"<p>No discussion found for ticker: {ticker.upper()}.</p>"
    pos_word_count = {}
    neg_word_count = {}
    for text in sub_df["combined_text"]:
        for token in text.lower().split():
            token_clean = token.strip(".,!?;:")
            if token_clean in pos_keywords:
                pos_word_count[token_clean] = pos_word_count.get(token_clean, 0) + 1
            if token_clean in neg_keywords:
                neg_word_count[token_clean] = neg_word_count.get(token_clean, 0) + 1

    sorted_pos = sorted(pos_word_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sorted_neg = sorted(neg_word_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def build_keywords_html(sorted_list):
        if not sorted_list:
            return "<p>None</p>"
        max_count = sorted_list[0][1]
        html = "<ul>"
        for i, (word, count) in enumerate(sorted_list, start=1):
            weight = round(count / max_count, 2)
            html += f"<li>{i}. <b>{word}</b> ({count}) - weight: {weight}</li>"
        html += "</ul>"
        return html

    pos_html = build_keywords_html(sorted_pos)
    neg_html = build_keywords_html(sorted_neg)
    return f"""
    <div class="keyword-container">
      <h2>{ticker.upper()} Keyword Ranking</h2>
      <h3>Top 10 Positive Keywords:</h3>
      {pos_html}
      <h3>Top 10 Negative Keywords:</h3>
      {neg_html}
    </div>
    """

def rank_stocks():
    ranking_results = []
    for company, ticker in company_map.items():
        sub_df = df[df['ticker'].str.upper() == ticker.upper()]
        if sub_df.empty:
            continue
        predictions = clf_pipeline.predict(sub_df["combined_text"])
        pos_count = (predictions == 1).sum()
        neg_count = (predictions == 0).sum()
        total = pos_count + neg_count
        if total == 0:
            continue
        net_score = (pos_count - neg_count) / total  
        ranking_results.append({
            "company": company,
            "ticker": ticker,
            "total_posts": total,
            "positive": pos_count,
            "negative": neg_count,
            "net_score": round(net_score, 2)
        })
    ranking_results = sorted(ranking_results, key=lambda x: x["net_score"], reverse=True)
    html = "<ol>"
    for r in ranking_results:
        html += f"<li><strong>{r['ticker']}</strong> ({r['company']}): {r['net_score']} [Total: {r['total_posts']}, +:{r['positive']}, -:{r['negative']}]</li>"
    html += "</ol>"
    return f"""
    <div class="ranking-container">
      <h2>Stock Ranking by Sentiment</h2>
      {html}
    </div>
    """
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

@app.route("/")
def home():
    return render_template("base.html", title="Stock Sentiment, Search & Keyword Ranking App")

@app.route("/ask")
def ask():
    question = request.args.get("question", "").strip()
    if not question:
        return jsonify({"response": "No question provided"}), 400
    ticker = map_company_to_ticker(question)
    if not ticker:
        return jsonify({"response": "Company not recognized"}), 400
    sentiment_html = analyze_sentiment_for_ticker(ticker)
    return jsonify({"response": sentiment_html})

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"response": "No search query provided"}), 400
    results_html = search_documents(query, top_n=10)
    return jsonify({"results": results_html})

@app.route("/keywords")
def keywords():
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return jsonify({"response": "No ticker provided"}), 400
    keywords_html = rank_keywords_for_ticker(ticker, top_n=10)
    return jsonify({"response": keywords_html})

@app.route("/rank")
def rank():
    ranking_html = rank_stocks()
    return jsonify({"response": ranking_html})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
