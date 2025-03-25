from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from preprocessing import custom_tokenizer

def create_classification_pipeline():
    """Create the sentiment classification pipeline."""
    # we are now logistical regression and TF-IDF instead of Naives Bayes for our classification
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english')),
        ('clf', LogisticRegression())
    ])

def create_ranking_pipeline():
    """Create the document ranking pipeline."""
    # we are now ranking with the help of TF-IDF and SVD instead of PCA
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english')),
        ('svd', TruncatedSVD(n_components=50))
    ])

def compute_cosine(v1, v2):
    """Compute cosine similarity between two vectors."""
    return sk_cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]