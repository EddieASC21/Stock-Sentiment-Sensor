from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from data_loader import load_data, map_company_to_ticker
from preprocessing import weak_label
from models import create_classification_pipeline, create_ranking_pipeline
from analysis import SentimentAnalyzer

app = Flask(__name__)
CORS(app)

# Load and preprocess data
df = load_data()
df["label"] = df["combined_text"].apply(weak_label)
df = df.dropna(subset=["label"])

# Create and fit pipelines
clf_pipeline = create_classification_pipeline()
clf_pipeline.fit(df["combined_text"], df["label"])

rank_pipeline = create_ranking_pipeline()
rank_vectors = rank_pipeline.fit_transform(df["combined_text"])
df["rank_vector"] = list(rank_vectors)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer(df, clf_pipeline, rank_pipeline)

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
    sentiment_html = sentiment_analyzer.analyze_sentiment_for_ticker(ticker)
    return jsonify({"response": sentiment_html})

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"response": "No search query provided"}), 400
    results_html = sentiment_analyzer.search_documents(query, top_n=10)
    return jsonify({"results": results_html})

@app.route("/keywords")
def keywords():
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return jsonify({"response": "No ticker provided"}), 400
    keywords_html = sentiment_analyzer.rank_keywords_for_ticker(ticker, top_n=10)
    return jsonify({"response": keywords_html})

@app.route("/rank")
def rank():
    ranking_html = sentiment_analyzer.rank_stocks()
    return jsonify({"response": ranking_html})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
