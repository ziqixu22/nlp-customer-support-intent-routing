from __future__ import annotations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_tfidf_logreg() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, solver="lbfgs")),
    ])
