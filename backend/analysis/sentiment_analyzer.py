import spacy
import numpy as np
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
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

def compute_market_statistics(ticker: str) -> dict:
    end = datetime.now()
    start = end - timedelta(days=30)
    hist = yf.Ticker(ticker).history(start=start, end=end)
    if hist.empty:
        return {}
    hist["ret"] = hist["Close"].pct_change().fillna(0)
    mean_ret = hist["ret"].mean()
    vol = hist["ret"].std()
    cum_return = hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
    cumprod = (1 + hist["ret"]).cumprod()
    running_max = cumprod.cummax()
    drawdowns = (cumprod - running_max) / running_max
    max_dd = drawdowns.min()
    return {
        "Mean Daily Return": mean_ret,
        "Daily Volatility": vol,
        "1M Cumulative Return": cum_return,
        "Max Drawdown": max_dd
    }

class SentimentAnalyzer:
    def __init__(self, df, clf_pipeline, rank_pipeline):
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
        self.vectorizer = self.clf_pipeline.named_steps["tfidf"]
        self.classifier = self.clf_pipeline.named_steps["clf"]
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.coeffs = self.classifier.coef_[0]

    def _compute_final_predictions(self, sub_df):
        base_preds = self.clf_pipeline.predict(sub_df["combined_text"])
        final_preds = base_preds.copy()
        for idx in range(len(sub_df)):
            text = sub_df.iloc[idx]["combined_text"]
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if sentences and min(self.sid.polarity_scores(s)["compound"] for s in sentences) < -0.05:
                final_preds[idx] = 0
            post_id = sub_df.iloc[idx].get("id", f"post_{idx}")
            up, down = get_vote_counts(post_id)
            if down >= up + 5:
                final_preds[idx] = 0
            elif up >= down + 3:
                final_preds[idx] = 1
        return final_preds

    def _get_feedback_stats_for_ticker(self, ticker: str):
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

    def _render_market_report(self, ticker: str) -> str:
        tkr = yf.Ticker(ticker)
        try:
            fast = tkr.fast_info
            price = fast.get("lastPrice", np.nan)
            prev = fast.get("previousClose", np.nan)
            day_low = fast.get("dayLow", np.nan)
            day_high = fast.get("dayHigh", np.nan)
            if np.isfinite(price) and np.isfinite(prev) and prev != 0:
                trend_pct = (price - prev) / prev * 100
                trend_str = f"{trend_pct:+.2f}%"
            else:
                trend_str = "N/A"
            stats = compute_market_statistics(ticker)
            mean_str = f"{stats.get('Mean Daily Return',0):.2%}"
            vol_str = f"{stats.get('Daily Volatility',0):.2%}"
            cum_str = f"{stats.get('1M Cumulative Return',0):.2%}"
            dd_str = f"{stats.get('Max Drawdown',0):.2%}"
            fin = tkr.get_financials(as_dict=True) or {}
            if fin:
                latest = max(fin.keys())
                rev = fin[latest].get("TotalRevenue")
                ni = fin[latest].get("NetIncome")
                rev_str = f"${rev:,.0f}" if isinstance(rev,(int,float)) else "N/A"
                ni_str = f"${ni:,.0f}" if isinstance(ni,(int,float)) else "N/A"
                date_str = latest.date().isoformat()
            else:
                rev_str = ni_str = date_str = "N/A"
            return f"""
        <div class="market-report">
        <h2 class="mr-title">{ticker.upper()} Market Report</h2>
        <section class="mr-price">
            <h3>Price & Trend</h3>
            <p><strong>Current:</strong> ${price:,.2f} &nbsp;|&nbsp; <strong>Prev Close:</strong> ${prev:,.2f}</p>
            <p><strong>Trend (1d):</strong> {trend_str}</p>
            <p><strong>Day Range:</strong> ${day_low:,.2f} – ${day_high:,.2f}</p>
        </section>
        <section class="mr-stats">
            <h3>1-Month Statistics</h3>
            <ul>
            <li><strong>Mean Daily Return:</strong> {mean_str}</li>
            <li><strong>Daily Volatility:</strong> {vol_str}</li>
            <li><strong>Cumulative Return (30d):</strong> {cum_str}</li>
            <li><strong>Max Drawdown:</strong> {dd_str}</li>
            </ul>
        </section>
        <section class="mr-financials">
            <h3>Latest Annual Financials ({date_str})</h3>
            <ul>
            <li><strong>Total Revenue:</strong> {rev_str}</li>
            <li><strong>Net Income:</strong> {ni_str}</li>
            </ul>
        </section>
        </div>
        """
        except Exception as e:
            return f"<p>Error fetching market data for <b>{ticker.upper()}</b>: {e}</p>"

    def analyze_sentiment_for_ticker(self, ticker: str, intent: str = None, days: int = 0) -> str:
        intent = (intent or "").lower()
        sub_df = self.df[self.df["ticker"].str.upper() == ticker.upper()]
        if days and not sub_df.empty:
            cutoff = datetime.now() - timedelta(days=days)
            sub_df = sub_df[sub_df["created_dt"] >= cutoff]
        if sub_df.empty:
            return self._render_market_report(ticker)
        final_preds = self._compute_final_predictions(sub_df)
        sub_df = sub_df.copy()
        sub_df["final_pred"] = final_preds
        pos_count = (sub_df["final_pred"] == 1).sum()
        neg_count = (sub_df["final_pred"] == 0).sum()
        total = pos_count + neg_count
        if total == 0:
            return self._render_market_report(ticker)
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
                post_class   = "Positive" if row["final_pred"] == 1 else "Negative"
                full_text    = row["combined_text"].strip()
                highlighted  = highlight_top_words(full_text, set(self.feature_names))
                orig_score   = row.get("score")
                score_html   = (
                    f"<p><strong>Original Reddit Score:</strong> {orig_score}</p>"
                    if orig_score is not None else ""
                )
                post_id      = row.get("id", f"post_{i}")
                up, down     = get_vote_counts(post_id)
                vote_html    = (
                    f"{score_html}"
                    f"<p>Upvotes: {up} | Downvotes: {down}</p>"
                    f"<button onclick=\"votePost('{post_id}','up')\">Upvote</button> "
                    f"<button onclick=\"votePost('{post_id}','down')\">Downvote</button>"
                )
                explanation_msg    = explain_post_sentiment(full_text)
                explanation_div_id = f"impact_{post_id}"
                explanation_html   = (
                    f"<a href='javascript:void(0);' onclick=\"toggleImpact('{explanation_div_id}')\">"
                    "[Show Impact]</a>"
                    f"<div id='{explanation_div_id}' style='display:none;margin-top:8px;'>"
                    f"{explanation_msg}</div>"
                )
                url = row.get("url","").strip()
                url_html = f"<br><a href='{url}' target='_blank'>Reddit Post</a>" if url else ""
                posts_html += (
                    f"<li><strong>{post_class}</strong> – {highlighted}"
                    f"{url_html}<br>{vote_html}<br>{explanation_html}</li>"
                )
            posts_html += "</ol>"
        feedback_html = self.get_feedback_for_ticker(ticker)
        return f"""
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

    def get_feedback_for_ticker(self, ticker: str) -> str:
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
        query = _summarize_query(query)
        query_vec = self.rank_pipeline.transform([query])[0]
        sims = [compute_cosine(query_vec, np.array(r)) for r in filtered_df["rank_vector"]]
        filtered_df = filtered_df.assign(sim_score=sims).sort_values("sim_score", ascending=False)
        top_df = filtered_df.head(top_n)
        posts = []
        for _, row in top_df.iterrows():
            pid = row.get("id", "unknown")
            text = row["combined_text"].strip()
            posts.append({
                "id": pid,
                "ticker": row["ticker"],
                "highlighted_text": highlight_top_words(text, set(self.feature_names)),
                "explanation": explain_post_sentiment(text),
                "url": row.get("url", ""),
                "score": float(row["sim_score"]),
                "upvotes": get_vote_counts(pid)[0],
                "downvotes": get_vote_counts(pid)[1],
            })
        overview = ""
        if ticker:
            total = len(filtered_df)
            final_preds = self._compute_final_predictions(filtered_df)
            pos = (final_preds == 1).sum()
            neg = (final_preds == 0).sum()
            tone = "Positive" if pos > neg else "Negative" if neg > pos else "Mixed"
            overview = (
                f"<h2>{ticker.upper()} – Top {len(top_df)} Posts</h2>"
                f"<p><strong>Overall Sentiment:</strong> {tone}</p>"
                f"<p><strong>Total:</strong> {total}, Positive: {pos}, Negative: {neg}</p>"
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
