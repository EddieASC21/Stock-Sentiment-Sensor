import numpy as np

from preprocessing import pos_keywords, neg_keywords, highlight_top_words, explain_post_sentiment
from models import compute_cosine
from data_loader import company_map

class SentimentAnalyzer:
  def __init__(self, df, clf_pipeline, rank_pipeline):
      self.df = df
      self.clf_pipeline = clf_pipeline
      self.rank_pipeline = rank_pipeline

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
        results_html += f"<li><strong>{row['ticker']}</strong> - {row['title']}<br><em>Score: {round(row['sim_score'],2)}</em></li>"
    results_html += "</ol>"
    return results_html

  def rank_keywords_for_ticker(self, ticker: str, top_n: int = 10):
      sub_df = self.df[self.df['ticker'].str.upper() == ticker.upper()]
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

  def rank_stocks(self):
      ranking_results = []
      for company, ticker in company_map.items():
          sub_df = self.df[self.df['ticker'].str.upper() == ticker.upper()]
          if sub_df.empty:
              continue
          predictions = self.clf_pipeline.predict(sub_df["combined_text"])
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

  def analyze_sentiment_for_ticker(self, ticker: str) -> str:
    sub_df = self.df[self.df['ticker'].str.upper() == ticker.upper()]
    if sub_df.empty:
        return f"<p>No recent discussion found for ticker: <b>{ticker.upper()}</b>.</p>"
    
    predictions = self.clf_pipeline.predict(sub_df["combined_text"])
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