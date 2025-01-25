#!/usr/bin/env python3
"""
Task 1
"""
from collections import Counter
import numpy as np


def generate_ngrams(sentence, n):
    """
    This function generates n-grams from a sentence.
    """
    return [" ".join(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]


def ngram_bleu(references, sentence, n):
    """
    This function calculates the n-gram BLEU score for a sentence.
    """
    candidate_ngrams = generate_ngrams(sentence, n)
    candidate_counts = Counter(candidate_ngrams)

    reference_ngrams = [generate_ngrams(ref, n) for ref in references]

    clipped_count = 0
    total_ngrams = len(candidate_ngrams)

    for ngram in candidate_counts:
        max_ref_count = max(ref.count(ngram) for ref in reference_ngrams)
        clipped_count += min(candidate_counts[ngram], max_ref_count)

    precision = clipped_count / total_ngrams if total_ngrams > 0 else 0

    c = len(sentence)
    r = min(references, key=lambda ref: abs(len(ref) - c))
    r = len(r)

    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r / c)

    bleu_score = BP * precision

    return bleu_score
