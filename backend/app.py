'''
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("StockNewsSentimentApp") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Optional: suppress Spark logs for cleanliness
spark.sparkContext.setLogLevel("WARN")

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
json_file_path = os.path.join(current_directory, 'init.json')


# Load JSON into PySpark DataFrame
df = spark.read.json(json_file_path)

# Assuming your JSON data is stored in a file named 'init.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)
    episodes_df = pd.DataFrame(data['episodes'])
    reviews_df = pd.DataFrame(data['reviews'])

with open(json_file_path, 'r') as file:
    data = json.load(file)

    # If data is a list, extract individual records
    if isinstance(data, list):
        df = pd.DataFrame(data)  # Convert list of dictionaries to DataFrame
    else:
        df = pd.DataFrame([data])  # Convert single dictionary to DataFrame

app = Flask(__name__)
CORS(app)


# Sample search using json with pandas
def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json


def json_search(query):
    # Filter rows that match the query in 'title' or 'text' column
    matches = df[df['title'].str.lower().str.contains(query.lower(), na=False) |
                 df['text'].str.lower().str.contains(query.lower(), na=False)]

    # Only keep columns that exist in the DataFrame
    expected_columns = ['title', 'text', 'score', 'flair', 'ticker', 'created_utc']
    existing_columns = [col for col in expected_columns if col in matches.columns]

    results = matches[existing_columns].copy()
    return results.to_json(orient='records')




@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    query = request.args.get("title", "")
    return json_search(query)



if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)'
'''
import os
import json
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import sparknlp
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower, concat_ws
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.feature import CountVectorizer, IDF  

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel
from pyspark.sql.functions import lower as spark_lower
from pyspark.ml.feature import PCA as PCA_Rank



app = Flask(__name__)
CORS(app)

spark = sparknlp.start()
spark.sparkContext.setLogLevel("WARN")

current_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(current_dir, "init.json")

with open(json_file_path, 'r') as infile:
    raw_data = json.load(infile)

# Convert JSON to Pandas DataFrame then to Spark DataFrame.
df = pd.DataFrame(raw_data if isinstance(raw_data, list) else [raw_data])
df['ticker'] = df.get('ticker', 'NONE')
df['ticker'] = df['ticker'].fillna('NONE')
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

spark_df = spark.createDataFrame(df)
# Create a combined text column for processing.
spark_df = spark_df.withColumn("combined_text", concat_ws(" ", col("title"), col("text")))

# would make sense to have a dataset with words that are positive and negative as it will provide a larger sample
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


spark_df = spark_df.withColumn(
    "label",
    when(spark_lower(col("combined_text")).rlike("|".join(pos_keywords)), 1)
    .when(spark_lower(col("combined_text")).rlike("|".join(neg_keywords)), 0)
    .otherwise(None)
)
spark_df = spark_df.dropna(subset=["label"])
spark_df = spark_df.withColumn("label", col("label").cast(IntegerType()))

document_assembler = DocumentAssembler()\
    .setInputCol("combined_text")\
    .setOutputCol("document")
tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")
normalizer = Normalizer()\
    .setInputCols(["token"])\
    .setOutputCol("normalized")\
    .setLowercase(True)
stopwords_cleaner = StopWordsCleaner()\
    .setInputCols(["normalized"])\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)
lemmatizer = LemmatizerModel.pretrained("lemma_antbnc", "en")\
    .setInputCols(["cleanTokens"])\
    .setOutputCol("lemma")
finisher = Finisher()\
    .setInputCols(["lemma"])\
    .setOutputCols(["finished_lemma"])\
    .setIncludeMetadata(False)
vectorizer = CountVectorizer(
    inputCol="finished_lemma",
    outputCol="countVec",
    vocabSize=2000,
    minDF=2
)
idf = IDF(
    inputCol="countVec",
    outputCol="tfidfVec"
)
nb = NaiveBayes(
    featuresCol="tfidfVec",
    labelCol="label",
    modelType="multinomial"
)

nlp_pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    normalizer,
    stopwords_cleaner,
    lemmatizer,
    finisher,
    vectorizer,
    idf,
    nb
])
sentiment_model = nlp_pipeline.fit(spark_df)


document_assembler_rank = DocumentAssembler()\
    .setInputCol("combined_text")\
    .setOutputCol("document")
tokenizer_rank = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")
normalizer_rank = Normalizer()\
    .setInputCols(["token"])\
    .setOutputCol("normalized")\
    .setLowercase(True)
stopwords_cleaner_rank = StopWordsCleaner()\
    .setInputCols(["normalized"])\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)
lemmatizer_rank = LemmatizerModel.pretrained("lemma_antbnc", "en")\
    .setInputCols(["cleanTokens"])\
    .setOutputCol("lemma")
finisher_rank = Finisher()\
    .setInputCols(["lemma"])\
    .setOutputCols(["finished_lemma"])\
    .setIncludeMetadata(False)
vectorizer_rank = CountVectorizer(
    inputCol="finished_lemma",
    outputCol="countVec",
    vocabSize=2000,
    minDF=2
)
idf_rank = IDF(
    inputCol="countVec",
    outputCol="tfidfVec"
)
pca_rank = PCA_Rank(
    k=50,
    inputCol="tfidfVec",
    outputCol="rankingFeatures"
)
ranking_pipeline = Pipeline(stages=[
    document_assembler_rank,
    tokenizer_rank,
    normalizer_rank,
    stopwords_cleaner_rank,
    lemmatizer_rank,
    finisher_rank,
    vectorizer_rank,
    idf_rank,
    pca_rank
])
ranking_model = ranking_pipeline.fit(spark_df)
ranked_df = ranking_model.transform(spark_df).cache()

# helper function
def cosine_similarity(v1, v2):
    arr1 = np.array(v1.toArray())
    arr2 = np.array(v2.toArray())
    dot = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def search_documents(query: str, top_n: int = 10):
    query_df = spark.createDataFrame([[query]], ["combined_text"])
    query_transformed = ranking_model.transform(query_df)
    query_vector = query_transformed.select("rankingFeatures").first()["rankingFeatures"]

    docs = ranked_df.select("title", "text", "combined_text", "rankingFeatures").collect()
    results = []
    for row in docs:
        doc_vector = row["rankingFeatures"]
        score = cosine_similarity(query_vector, doc_vector)
        results.append({
            "title": row["title"],
            "text": row["text"],
            "combined_text": row["combined_text"],
            "similarity": score
        })
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]

def rank_keywords_for_ticker(ticker: str, top_n: int = 10):
    sub = df[df['ticker'].str.upper() == ticker.upper()]
    if sub.empty:
        return f"No discussion found for ticker: {ticker.upper()}."
    relevant_texts = (sub['title'] + " " + sub['text']).str.lower().tolist()
    pos_word_count = {}
    neg_word_count = {}
    for doc in relevant_texts:
        tokens = doc.split()
        for w in tokens:
            if w in pos_keywords:
                pos_word_count[w] = pos_word_count.get(w, 0) + 1
            if w in neg_keywords:
                neg_word_count[w] = neg_word_count.get(w, 0) + 1
    sorted_pos = sorted(pos_word_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sorted_neg = sorted(neg_word_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if sorted_pos:
        max_pos = sorted_pos[0][1]
        pos_ranked = [
            {"word": word, "count": count, "weight": round(count / max_pos, 2)}
            for word, count in sorted_pos
        ]
    else:
        pos_ranked = []
    if sorted_neg:
        max_neg = sorted_neg[0][1]
        neg_ranked = [
            {"word": word, "count": count, "weight": round(count / max_neg, 2)}
            for word, count in sorted_neg
        ]
    else:
        neg_ranked = []
    return {"positive": pos_ranked, "negative": neg_ranked}

# we should have a json file with names to tickers
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


def explain_post_sentiment(post_text: str) -> str:
    tokens = post_text.lower().split()
    pos_found = [t for t in tokens if t.strip(".,!?;:") in pos_keywords]
    neg_found = [t for t in tokens if t.strip(".,!?;:") in neg_keywords]
    pos_unique = list(set(pos_found))
    neg_unique = list(set(neg_found))
    explanation = ""
    if pos_unique:
        explanation += "Positive keywords: " + ", ".join(pos_unique)
    if neg_unique:
        if explanation:
            explanation += " | "
        explanation += "Negative keywords: " + ", ".join(neg_unique)
    if not explanation:
        explanation = "No significant sentiment keywords found."
    return explanation


def analyze_sentiment_for_ticker(ticker: str) -> str:
    sub = df[df['ticker'].str.upper() == ticker.upper()]
    if sub.empty:
        return f"<p>No recent discussion found for ticker: <b>{ticker.upper()}</b>.</p>"
    
    sdf = spark.createDataFrame(sub).withColumn("combined_text", concat_ws(" ", col("title"), col("text")))
    preds = sentiment_model.transform(sdf)
    pdf = preds.select("title", "text", "prediction").toPandas()
    
    pos_count = (pdf["prediction"] == 1).sum()
    neg_count = (pdf["prediction"] == 0).sum()
    total = pos_count + neg_count
    if total == 0:
        return f"<p>Sentiment about <b>{ticker.upper()}</b> is unclear or neutral.</p>"
    
    tone = "Positive" if pos_count > neg_count else "Negative" if neg_count > pos_count else "Mixed"

    relevant_texts = (sub['title'] + " " + sub['text']).str.lower().tolist()
    pos_word_count = {}
    neg_word_count = {}
    for doc in relevant_texts:
        tokens = doc.split()
        for w in tokens:
            token = w.strip(".,!?;:")
            if token in pos_keywords:
                pos_word_count[token] = pos_word_count.get(token, 0) + 1
            if token in neg_keywords:
                neg_word_count[token] = neg_word_count.get(token, 0) + 1
    sorted_pos = sorted(pos_word_count.items(), key=lambda x: x[1], reverse=True)[:10]
    sorted_neg = sorted(neg_word_count.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if sorted_pos:
        max_pos = sorted_pos[0][1]
        pos_html = "<ul>"
        for i, (word, count) in enumerate(sorted_pos, start=1):
            weight = round(count / max_pos, 2)
            pos_html += f"<li>{i}. <b>{word}</b> ({count}) - weight: {weight}</li>"
        pos_html += "</ul>"
    else:
        pos_html = "<p>None</p>"

    if sorted_neg:
        max_neg = sorted_neg[0][1]
        neg_html = "<ul>"
        for i, (word, count) in enumerate(sorted_neg, start=1):
            weight = round(count / max_neg, 2)
            neg_html += f"<li>{i}. <b>{word}</b> ({count}) - weight: {weight}</li>"
        neg_html += "</ul>"
    else:
        neg_html = "<p>None</p>"

    posts_html = "<ol>"
    top_5_posts = pdf.head(5)
    for idx, row in top_5_posts.iterrows():
        classification = "Positive" if row["prediction"] == 1 else "Negative"
        full_text = (row["title"] + " " + row["text"]).strip()

        def highlight_top_words(text: str) -> str:
            tokens = text.split()
            new_tokens = []
            for t in tokens:
                t_clean = t.lower().strip(".,!?;:")
                if t_clean in {w for w, _ in sorted_pos} or t_clean in {w for w, _ in sorted_neg}:
                    new_tokens.append(f"<b>{t}</b>")
                else:
                    new_tokens.append(t)
            return " ".join(new_tokens)
        highlighted_text = highlight_top_words(full_text)
        explanation = explain_post_sentiment(full_text)
        posts_html += f"<li><strong>{classification}</strong> - {highlighted_text}<br><em>{explanation}</em></li>"
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


@app.route("/rank")
def rank_stocks():
    ranking_results = []
    for company, ticker in company_map.items():
        sub = df[df['ticker'].str.upper() == ticker.upper()]
        if sub.empty:
            continue
        sdf = spark.createDataFrame(sub).withColumn("combined_text", concat_ws(" ", col("title"), col("text")))
        preds = sentiment_model.transform(sdf)
        pdf = preds.groupBy("prediction").count().toPandas()
        pos_count = pdf.loc[pdf["prediction"] == 1, "count"].sum() if 1 in pdf["prediction"].values else 0
        neg_count = pdf.loc[pdf["prediction"] == 0, "count"].sum() if 0 in pdf["prediction"].values else 0
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
    return jsonify({"ranking": ranking_results})

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
    sentiment_paragraph = analyze_sentiment_for_ticker(ticker)
    return jsonify({"response": sentiment_paragraph})

@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"response": "No search query provided"}), 400
    results = search_documents(query, top_n=10)
    return jsonify({"results": results})

@app.route("/keywords")
def keywords():
    ticker = request.args.get("ticker", "").strip()
    if not ticker:
        return jsonify({"response": "No ticker provided"}), 400
    ranked_keywords = rank_keywords_for_ticker(ticker, top_n=10)
    return jsonify({"ticker": ticker.upper(), "ranked_keywords": ranked_keywords})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
