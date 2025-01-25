#!/usr/bin/env python3
"""
Task 0
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    This function calculates the unigram BLEU score for a sentence.
    """
    candidate_counts = {}
    for word in sentence:
        candidate_counts[word] = candidate_counts.get(word, 0) + 1

    clipped_count = 0
    total_unigrams = len(sentence)

    for word in candidate_counts:
        max_ref_count = max(ref.count(word) for ref in references)
        clipped_count += min(candidate_counts[word], max_ref_count)

    precision = clipped_count / total_unigrams

    c = len(sentence)
    r = min(references, key=lambda ref: abs(len(ref) - c))
    r = len(r)

    if c > r:
        BP = 1
    else:
        BP = np.exp(1 - r / c)

    bleu_score = BP * precision

    return bleu_score
