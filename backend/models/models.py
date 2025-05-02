from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from preprocessing.text_processor import custom_tokenizer

def create_classification_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', OneVsRestClassifier(LogisticRegression(
            solver='liblinear',
            max_iter=1000,
            C=1.0,
            class_weight='balanced'
        )))
    ])

def create_ranking_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english')),
        ('svd', TruncatedSVD(n_components=50))
    ])

def compute_cosine(v1, v2):
    return sk_cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]
