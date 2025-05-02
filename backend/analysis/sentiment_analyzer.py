import spacy
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.text_processor import custom_tokenizer, highlight_top_words, explain_post_sentiment
from helpers.vote_helper import get_vote_counts

nlp_spacy = spacy.load("en_core_web_sm")

def _summarize_query(query: str) -> str:
    if not query:
        return query
    if len(query.split()) > 8:
        return query
    doc = nlp_spacy(query)
    keywords = set()
    for token in doc:
        if not token.is_stop and token.is_alpha and token.pos_ in ["NOUN", "ADJ", "VERB"]:
            keywords.add(token.lemma_)
    if len(keywords) < len(query.split()):
        return " ".join(keywords)
    else:
        return query

class SentimentAnalyzer:
    def __init__(self, df, clf_pipeline, rank_pipeline, vectorizer):
        self.df = df.copy()
        self.clf_pipeline = clf_pipeline
        self.rank_pipeline = rank_pipeline
        self.sid = SentimentIntensityAnalyzer()
        if "timestamp" in self.df.columns:
            self.df["created_dt"] = pd.to_datetime(self.df["timestamp"], unit="s", errors="coerce")
        elif "created" in self.df.columns:
            self.df["created_dt"] = pd.to_datetime(self.df["created"], errors="coerce")
        else:
            self.df["created_dt"] = pd.Timestamp.now()
        self.vectorizer = vectorizer
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _compute_final_predictions(self, sub_df):
        # Get base predictions as strings
        base_preds = self.clf_pipeline.predict(sub_df["combined_text"])
        final_preds = base_preds.copy()
        
        # Create a mapping for string labels to numbers
        label_to_num = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }
        
        # Reverse mapping to convert back to strings
        num_to_label = {
            0: 'negative',
            1: 'neutral',
            2: 'positive'
        }
        
        for idx in range(len(sub_df)):
            text = sub_df.iloc[idx]["combined_text"]
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            current_pred = label_to_num[base_preds[idx]]
            
            if sentences and min(self.sid.polarity_scores(s)["compound"] for s in sentences) < -0.05:
                current_pred = 0  # Set to negative
            
            # Apply vote count rules
            post_id = sub_df.iloc[idx].get("id", f"post_{idx}")
            up, down = get_vote_counts(post_id)
            
            if down >= up + 5:
                current_pred = 0  # Set to negative
            elif up >= down + 3:
                current_pred = 2  # Set to positive
            
            # Convert back to string label
            final_preds[idx] = num_to_label[current_pred]
        
        return final_preds

    def _get_feedback_stats_for_ticker(self, ticker: str):
        if ticker is None or ticker == "":
            return None, 0
        feedback_file = os.path.join(os.path.dirname(__file__), "../feedback.csv")
        if not os.path.exists(feedback_file):
            return None, 0
        try:
            feedback_df = pd.read_csv(feedback_file)
        except Exception:
            return None, 0
        filtered = feedback_df[feedback_df["ticker"].str.upper() == ticker.upper()]
        if filtered.empty:
            return None, 0
        avg_rating = filtered["rating"].mean()
        count = len(filtered)
        return avg_rating, count

    def get_feedback_for_ticker(self, ticker: str) -> str:
        if ticker is None or ticker == "":
            return "<p>No user feedback available for this stock.</p>"
        feedback_file = os.path.join(os.path.dirname(__file__), "../feedback.csv")
        if not os.path.exists(feedback_file):
            return "<p>No user feedback available for this stock.</p>"
        try:
            feedback_df = pd.read_csv(feedback_file)
        except Exception:
            return "<p>No user feedback available for this stock.</p>"
        filtered = feedback_df[feedback_df["ticker"].str.upper() == ticker.upper()]
        if filtered.empty:
            return "<p>No user feedback available for this stock.</p>"
        avg_rating = filtered["rating"].mean()
        feedback_html = f"<p><strong>Average Rating:</strong> {round(avg_rating, 2)}</p><ol>"
        for idx, row in filtered.iterrows():
            feedback_html += (
                f"<li><strong>Rating:</strong> {row['rating']} - <em>{row['comments']}</em> (<small>{row['timestamp']}</small>)</li>"
            )
        feedback_html += "</ol>"
        return f"<div class='feedback-display'><h3>User Feedback</h3>{feedback_html}</div>"

    def search_comments(self, query: str, ticker: str=None, intent: str=None, top_n: int=100) -> dict:
        filtered_df = self.df.copy()
        if ticker:
            filtered_df = filtered_df[filtered_df["ticker"].str.upper() == ticker.upper()]
            if filtered_df.empty:
                filtered_df = self.df.copy()
        consice_query = _summarize_query(query)
        query_vec = self.rank_pipeline.transform([consice_query])[0]
        rank_vectors = np.stack(filtered_df["rank_vector"].values)
        sims = cosine_similarity(query_vec.reshape(1, -1), rank_vectors).flatten()
        filtered_df = filtered_df.assign(sim_score=sims).sort_values("sim_score", ascending=False)
        top_df = filtered_df.head(top_n)
        posts = []
        for idx, row in top_df.iterrows():
            post_id_raw = row.get("id")
            if pd.isna(post_id_raw):
                post_id = str(idx)
            else:
                post_id = str(post_id_raw) # Ensure it's a string
            if not post_id:
                print(f"Invalid post ID for row {idx}: {row}")
                continue
            text = row["combined_text"].strip()
            posts.append({
                "id": post_id,
                "ticker": row["ticker"],
                "highlighted_text": highlight_top_words(text, set(self.feature_names)),
                "explanation": explain_post_sentiment(text, self.clf_pipeline),
                "url": row.get("url", ""),
                "sentiment": row.get("label", "unknown"),
                "score": float(row["sim_score"]),
                "upvotes": get_vote_counts(post_id)[0],
                "downvotes": get_vote_counts(post_id)[1],
            })
        overview = ""
        if ticker:
            total = len(filtered_df)
            final_preds = self._compute_final_predictions(filtered_df)
            pos = sum(pred == 'positive' for pred in final_preds)
            neut = sum(pred == 'neutral' for pred in final_preds)
            neg = sum(pred == 'negative' for pred in final_preds)
            
            pos_percent = (pos / total) * 100 if total > 0 else 0
            neg_percent = (neg / total) * 100 if total > 0 else 0
            if pos > neg * 2 and pos_percent > 60:
                tone = "Strongly Positive"
            elif pos > neg and pos_percent > 40:
                tone = "Positive"
            elif neg > pos * 2 and neg_percent > 60:
                tone = "Strongly Negative"
            elif neg > pos and neg_percent > 40:
                tone = "Negative"
            elif neut > (pos + neg):
                tone = "Mostly Neutral"
            else:
                tone = "Mixed"
            overview = (
                f"<h2>{ticker.upper()} – Top {len(top_df)} Posts</h2>"
                f"<p><strong>Overall Sentiment:</strong> {tone}</p>"
                f"<p><strong>Total:</strong> {total}, Positive: {pos}, Negative: {neg} and Neutral: {neut}</p>"
            )
        header_html = f"<div class='search-comments'>{overview}<h3>Search Results for “{query}”</h3>"
        fb_avg, fb_n = self._get_feedback_stats_for_ticker(ticker)
        footer_html = (
            "<div class='feedback-display'>"
            f"<p><strong>Avg Rating:</strong> {round(fb_avg,2) if fb_avg else 'N/A'} ({fb_n} reviews)</p></div></div>"
        )
        return {"header_html": header_html, "posts": posts, "footer_html": footer_html}

    def train_and_evaluate_for_ticker(self, ticker: str) -> str:
        sub_df = self.df[self.df["ticker"].str.upper() == ticker.upper()]
        if sub_df.empty:
            return f"<p>No data found for ticker: <b>{ticker.upper()}</b>.</p>"
        X, y = sub_df["combined_text"], sub_df["label"]
        if len(X) < 10:
            return f"<p>Not enough data for ticker: <b>{ticker.upper()}</b> to perform training/test evaluation (need at least 10 records).</p>"
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        eval_pipe = Pipeline([("tfidf", TfidfVectorizer(tokenizer=custom_tokenizer, stop_words="english")), ("clf", LogisticRegression())])
        eval_pipe.fit(X_train, y_train)
        y_pred = eval_pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return f"""
        <div class="evaluation-container">
          <h2>Training & Evaluation for {ticker.upper()}</h2>
          <p><strong>Test Accuracy:</strong> {round(acc, 2)}</p>
          <h3>Classification Report:</h3>
          <pre>{report}</pre>
        </div>
        """
