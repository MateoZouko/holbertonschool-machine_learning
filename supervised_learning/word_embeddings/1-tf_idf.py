#!/usr/bin/env python3
"""
Task 1
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix.
    """
    processed_sentences = []
    for sentence in sentences:
        words = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        processed_sentences.append(" ".join(words))

    if vocab is None:
        vocab = sorted(set(
            word for sentence in processed_sentences
            for word in sentence.split()))

    tf_idf_vect = TfidfVectorizer(vocabulary=vocab)

    tfidf_matrix = tf_idf_vect.fit_transform(processed_sentences)

    features = tf_idf_vect.get_feature_names_out()

    embeddings = tfidf_matrix.toarray()

    return embeddings, features
