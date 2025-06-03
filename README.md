# Stock Sentiment Sensor

A Flask-based web app that aggregates and analyzes public sentiment about stocks from multiple online sources. Users can search by keyword or ticker to see the most relevant posts, their inferred sentiment (positive/neutral/negative), and overall sentiment metrics. The app also supports upvoting/downvoting posts and submitting feedback per ticker.

**Live Demo:**  
http://4300showcase.infosci.cornell.edu:5262/

---

## What the Project Does

- **Data Ingestion & Cleaning**  
  Loads a JSON file of stock-related posts (e.g. Reddit threads, news headlines). Filters out short or irrelevant posts based on word count, finance keywords, and slang.

- **Sentiment Classification**  
  Trains a 3-class classifier (TF-IDF + Logistic Regression) on the Financial Phrasebank. Labels each post’s text as positive, neutral, or negative.

- **Similarity Search**  
  Converts every post into a 50-dim embedding (TF-IDF → SVD). When a user submits a query, the app returns the top posts by cosine similarity.

- **Aggregate Metrics**  
  For a given ticker, computes overall “tone” (e.g. Strongly Positive, Negative, Mostly Neutral) by counting and adjusting labels. Displays positive/neutral/negative counts and average user feedback.

- **Voting & Feedback**  
  Users can upvote/downvote individual posts (stored in a CSV) and submit a rating (1–5) plus comments for any ticker. Average feedback is shown alongside search results.

---

## Basic Usage

1. **Clone & install dependencies**  
   ```bash
   git clone https://github.com/EddieASC21/Stock-Sentiment-Sensor.git
   cd Stock-Sentiment-Sensor/backend
   python -m venv venv
   source venv/bin/activate   
   pip install -r requirements.txt

### Prepare `init.json`
Replace or extend `init.json` with your own JSON array of posts. Each entry needs:
```json
{
  "ticker": "AAPL",
  "title": "Apple Q4 Earnings Beat",
  "text": "Apple reported record revenue this quarter...",
  "url": "https://example.com/article"
}
```
## Run the server
```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

Open your browser at `http://localhost:5000/` (or use the live demo link above).

---

### Search
Visit:
```bash
/ask?question=your+query&ticker=TSLA&days=0
```
to retrieve up to 100 relevant posts with sentiment labels.

### Vote
POST to `/vote` with JSON:
```json
{ "post_id": "<id>", "vote": "up" }
```
or "down" to update vote counts.

### Vote
POST to `/feedback` with:
```json
{ "ticker": "TSLA", "rating": 4, "comments": "Great insights!" }
```
to save user feedback.

### Dependencies

- Flask  
- pandas, numpy  
- scikit-learn  
- spaCy (en_core_web_sm)  
- nltk (VADER)

Install all with:
```bash
pip install -r requirements.txt
```
