import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from models import compute_cosine
from data_loader import company_map
from preprocessing.text_processor import custom_tokenizer, weak_label, highlight_top_words, explain_post_sentiment
from helpers.vote_helper import get_vote_counts

class SentimentAnalyzer:
    def __init__(self, df, clf_pipeline, rank_pipeline):
        self.df = df
        self.clf_pipeline = clf_pipeline
        self.rank_pipeline = rank_pipeline
        self.sid = SentimentIntensityAnalyzer()
        self.vectorizer = self.clf_pipeline.named_steps['tfidf']
        self.classifier = self.clf_pipeline.named_steps['clf']
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.coeffs = self.classifier.coef_[0]

    def search_documents(self, query: str, top_n: int = 10):
        query_vec = self.rank_pipeline.transform([query])[0]
        similarities = []
        for idx, row in self.df.iterrows():
            doc_vec = row["rank_vector"]
            sim = compute_cosine(np.array(query_vec), np.array(doc_vec))
            similarities.append(sim)
        self.df["sim_score"] = similarities
        results = self.df.sort_values("sim_score", ascending=False).head(top_n)
        results_html = "<ol>"
        for _, row in results.iterrows():
            results_html += (
                f"<li><strong>{row['ticker']}</strong> - {row['title']}<br>"
                f"<em>Score: {round(row['sim_score'],2)}</em></li>"
            )
        results_html += "</ol>"
        return results_html

    def rank_stocks(self):
        final_preds = self._compute_final_predictions(self.df)
        self.df["final_pred"] = final_preds

        ranking_results = []
        for company, ticker in company_map.items():
            sub_df = self.df[self.df['ticker'].str.upper() == ticker.upper()]
            if sub_df.empty:
                continue
            pos_count = (sub_df["final_pred"] == 1).sum()
            neg_count = (sub_df["final_pred"] == 0).sum()
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

    def _compute_final_predictions(self, sub_df):
        base_preds = self.clf_pipeline.predict(sub_df["combined_text"])
        final_preds = base_preds.copy()
        for idx in range(len(sub_df)):
            text = sub_df.iloc[idx]["combined_text"]
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences and min(self.sid.polarity_scores(s)['compound'] for s in sentences) < -0.05:
                final_preds[idx] = 0
            post_id = sub_df.iloc[idx].get("id", f"post_{idx}")
            upvotes, downvotes = get_vote_counts(post_id)
            if downvotes >= upvotes + 5:
                final_preds[idx] = 0
            elif upvotes >= downvotes + 3:
                final_preds[idx] = 1
        return final_preds

    def _get_feedback_stats_for_ticker(self, ticker: str):
        feedback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../feedback.csv")
        if not os.path.exists(feedback_file):
            return None, 0
        try:
            feedback_df = pd.read_csv(feedback_file)
        except Exception:
            return None, 0
        filtered = feedback_df[feedback_df['ticker'].str.upper() == ticker.upper()]
        if filtered.empty:
            return None, 0
        avg_rating = filtered['rating'].mean()
        count = len(filtered)
        return avg_rating, count

    def analyze_sentiment_for_ticker(self, ticker: str, intent: str = None) -> str:
        sub_df = self.df[self.df['ticker'].str.upper() == ticker.upper()]
        if sub_df.empty:
            return f"<p>No recent discussion found for ticker: <b>{ticker.upper()}</b>.</p>"

        final_preds = self._compute_final_predictions(sub_df)
        sub_df = sub_df.copy()
        sub_df["final_pred"] = final_preds

        pos_count = (sub_df["final_pred"] == 1).sum()
        neg_count = (sub_df["final_pred"] == 0).sum()
        total = pos_count + neg_count
        if total == 0:
            return f"<p>Sentiment about <b>{ticker.upper()}</b> is unclear or neutral.</p>"
        overall_tone = "Positive" if pos_count > neg_count else ("Negative" if neg_count > pos_count else "Mixed")

        feedback_avg, feedback_count = self._get_feedback_stats_for_ticker(ticker)
        if feedback_count >= 3 and feedback_avg is not None and feedback_avg < 3.0:
            overall_tone = "Negative"

        if intent == "positive":
            custom_header = f"<h3>You asked about good sentiment on {ticker.upper()}. Out of {total} posts, there are {pos_count} positive.</h3>"
        elif intent == "negative":
            custom_header = f"<h3>You asked about bad sentiment on {ticker.upper()}. Out of {total} posts, there are {neg_count} negative.</h3>"
        else:
            custom_header = f"<h3>Based on our model, votes, and user feedback, the overall sentiment is {overall_tone.lower()} on {ticker.upper()}.</h3>"

        intent = intent.lower() if intent else ""
        if intent == "positive":
            filtered_df = sub_df[sub_df["final_pred"] == 1]
            display_note = "<p>Showing only <b>positive</b> posts:</p>"
        elif intent == "negative":
            filtered_df = sub_df[sub_df["final_pred"] == 0]
            display_note = "<p>Showing only <b>negative</b> posts:</p>"
        else:
            filtered_df = sub_df
            display_note = "<p>Showing all posts:</p>"

        if filtered_df.empty:
            posts_html = "<p>No posts found for the specified sentiment.</p>"
        else:
            posts_html = "<ol>"
            for i, row in filtered_df.head(5).iterrows():
                post_class = "Positive" if row["final_pred"] == 1 else "Negative"
                full_text = row["combined_text"].strip()
                highlighted_text = highlight_top_words(full_text, set(self.feature_names))
                
                explanation_msg = explain_post_sentiment(full_text)
                
                post_id = row.get("id", f"post_{i}")
                explanation_div_id = f"impact_{post_id}"
                explanation_html = f"""
                <a href="javascript:void(0);" onclick="toggleImpact('{explanation_div_id}')">[Show Impact]</a>
                <div id="{explanation_div_id}" style="display:none; margin-top:8px;">
                  {explanation_msg}
                </div>
                """
                url = row.get("url", "")
                url_html = f"<br><a href='{url}' target='_blank'>Reddit Post</a>" if url.strip() else ""
                upvotes, downvotes = get_vote_counts(post_id)
                vote_html = (f"<p>Upvotes: {upvotes} | Downvotes: {downvotes}</p>"
                             f"<button onclick=\"votePost('{post_id}','up')\">Upvote</button> "
                             f"<button onclick=\"votePost('{post_id}','down')\">Downvote</button>")
                posts_html += (f"<li><strong>{post_class}</strong> - {highlighted_text}{url_html}"
                               f"<br>{vote_html}<br>{explanation_html}</li>")
            posts_html += "</ol>"

        feedback_html = self.get_feedback_for_ticker(ticker)

        output_html = f"""
        <div class="sentiment-container">
          <h2>{ticker.upper()} Sentiment Analysis</h2>
          {custom_header}
          <p><strong>Total relevant posts:</strong> {total}</p>
          <p><strong>Overall Sentiment:</strong> {overall_tone}</p>
          <p><strong>Positive Mentions:</strong> {pos_count}</p>
          <p><strong>Negative Mentions:</strong> {neg_count}</p>
          {display_note}
          <h3>Top 5 Relevant Posts:</h3>
          {posts_html}
          {feedback_html}
        </div>
        """
        return output_html

    def get_feedback_for_ticker(self, ticker: str) -> str:
        feedback_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../feedback.csv")
        if not os.path.exists(feedback_file):
            return "<p>No user feedback available for this stock.</p>"
        try:
            feedback_df = pd.read_csv(feedback_file)
        except Exception:
            return "<p>No user feedback available for this stock.</p>"
        filtered = feedback_df[feedback_df['ticker'].str.upper() == ticker.upper()]
        if filtered.empty:
            return "<p>No user feedback available for this stock.</p>"
        avg_rating = filtered['rating'].mean()
        feedback_html = f"<p><strong>Average Rating:</strong> {round(avg_rating, 2)}</p>"
        feedback_html += "<ol>"
        for idx, row in filtered.iterrows():
            feedback_html += (f"<li><strong>Rating:</strong> {row['rating']} - <em>{row['comments']}</em> (<small>{row['timestamp']}</small>)</li>")
        feedback_html += "</ol>"
        return f"""
        <div class="feedback-display">
          <h3>User Feedback</h3>
          {feedback_html}
        </div>
        """

    def train_and_evaluate_for_ticker(self, ticker: str) -> str:
        sub_df = self.df[self.df['ticker'].str.upper() == ticker.upper()]
        if sub_df.empty:
            return f"<p>No data found for ticker: <b>{ticker.upper()}</b>.</p>"
        X = sub_df["combined_text"]
        y = sub_df["label"]
        if len(X) < 10:
            return f"<p>Not enough data for ticker: <b>{ticker.upper()}</b> to perform training/test evaluation (need at least 10 records).</p>"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        eval_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english')),
            ('clf', LogisticRegression())
        ])
        eval_pipeline.fit(X_train, y_train)
        y_pred = eval_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        output_html = f"""
        <div class="evaluation-container">
          <h2>Training & Evaluation for {ticker.upper()}</h2>
          <p><strong>Test Accuracy:</strong> {round(acc, 2)}</p>
          <h3>Classification Report:</h3>
          <pre>{report}</pre>
        </div>
        """
        return output_html
