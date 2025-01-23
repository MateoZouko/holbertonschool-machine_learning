#!/usr/bin/env python3
"""
Task 0
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix.
    """
    processed_sentences = []
    for sentence in sentences:
        sentence = re.findall(r'\b[a-zA-Z]{2,}\b', sentence.lower())
        processed_sentences.append(sentence)

    if vocab is None:
        vocab = sorted(set(
            word for sentence in processed_sentences for word in sentence))

    s = len(processed_sentences)
    f = len(vocab)
    embeddings = np.zeros((s, f), dtype=int)

    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    vocab = np.array(vocab)

    return embeddings, vocab
