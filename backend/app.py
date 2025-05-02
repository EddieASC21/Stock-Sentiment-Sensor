import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

nltk.download('vader_lexicon')

from data_loader import load_data, map_company_to_ticker, valid_ticker, load_financial_phrasebank
from preprocessing import clean_comments
from models import create_classification_pipeline, create_ranking_pipeline, highlight_tfidf_vectorizer
from analysis import SentimentAnalyzer

app = Flask(__name__)
CORS(app)

sid = SentimentIntensityAnalyzer()

df = load_data()
if "combined_text" not in df.columns:
    df["combined_text"] = df["title"] + " " + df["text"]
sentiment_df = load_financial_phrasebank('Sentences_75Agree.txt')
tfidf_vectorizer = highlight_tfidf_vectorizer()
tfidf_vectorizer.fit(df["combined_text"])

df = clean_comments(df)

# Train pipelines
clf_pipeline = create_classification_pipeline()
clf_pipeline.fit(sentiment_df["text"], sentiment_df["label"])

df['label'] = clf_pipeline.predict(df["combined_text"])
df = df.dropna(subset=["label"])

rank_pipeline = create_ranking_pipeline()
rank_vectors = rank_pipeline.fit_transform(df["combined_text"])
df["rank_vector"] = list(rank_vectors)

sentiment_analyzer = SentimentAnalyzer(df, clf_pipeline, rank_pipeline, tfidf_vectorizer)

@app.route("/")
def home():
    return render_template("base.html", title="Stock Sentiment, Search & Keyword Ranking App")


@app.route("/ask")
def ask():
    question = request.args.get("question", "").strip()
    if not question:
        return jsonify({"response": "No question provided"}), 400

    ticker_q = request.args.get("ticker", "").strip().upper()
    if ticker_q and valid_ticker(ticker_q):
        ticker = ticker_q
    else:
        ticker = map_company_to_ticker(question)

    try:
        days = int(request.args.get("days", "0"))
    except ValueError:
        days = 0

    scores = sid.polarity_scores(question)
    compound = scores["compound"]
    if compound >= 0.05:
        intent = "positive"
    elif compound <= -0.05:
        intent = "negative"
    else:
        intent = ""

    result = sentiment_analyzer.search_comments(
        question,
        ticker=ticker,
        intent=intent,
        top_n=100
    )

    return jsonify({
        "response":     "Data retrieved successfully",
        "header_html":  result["header_html"],
        "posts":        result["posts"],
        "footer_html":  result["footer_html"]
    })


@app.route("/vote", methods=["POST"])
def vote():
    data = request.get_json()
    post_id = data.get("post_id")
    vote_val = data.get("vote") or data.get("vote_type")
    if not post_id or vote_val not in ["up", "down"]:
        return jsonify({"response": "Invalid vote data."}), 400

    from helpers.vote_helper import update_vote, get_vote_counts
    new_counts = update_vote(post_id, vote_val)
    return jsonify({
        "response": f"New vote counts – Upvotes: {new_counts[0]}, Downvotes: {new_counts[1]}."
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    ticker = data.get("ticker", "").strip().upper()
    try:
        rating = float(data.get("rating", 3))
    except ValueError:
        return jsonify({"response": "Invalid rating value."}), 400
    comments = data.get("comments", "").strip()
    if not ticker or not rating:
        return jsonify({"response": "Ticker or rating missing."}), 400

    from datetime import datetime
    feedback_row = {
        "ticker":   ticker,
        "rating":   rating,
        "comments": comments,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    import pandas as pd
    feedback_file = os.path.join(os.path.dirname(__file__), "feedback.csv")
    if os.path.exists(feedback_file):
        existing = pd.read_csv(feedback_file)
        pd.concat([existing, pd.DataFrame([feedback_row])], ignore_index=True) \
          .to_csv(feedback_file, index=False)
    else:
        pd.DataFrame([feedback_row]).to_csv(feedback_file, index=False)

    return jsonify({"response": f"Feedback saved for {ticker} with rating={rating}."})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)