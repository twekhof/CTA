# cta.py
# calculate quality for all topics

import pandas as pd
import numpy as np
from itertools import combinations
from itertools import product


def calculate_quality_all(words_per_topic, cosine_matrix):
    quality_all_tmp = []
    topic_combinations = list(combinations(range(len(words_per_topic)), 2))
    for i in range(len(topic_combinations)):
        topic_k = topic_combinations[i][0]
        topic_l = topic_combinations[i][1]
        topic_k_words = words_per_topic[topic_k]
        topic_k_words_combinations = list(combinations(topic_k_words, 2))
        topic_l_words = words_per_topic[topic_l]
        topic_l_words_combinations = list(combinations(topic_l_words, 2))

        # coherence_k
        tmp_intra_topic_k = list(combinations(topic_k_words, 2))
        coherence_k = []
        for j in range(len(tmp_intra_topic_k)):
            coherence_tmp = cosine_matrix.loc[topic_k_words_combinations[j][1], topic_k_words_combinations[j][0]]
            coherence_k.append(coherence_tmp)
        coherence_k = np.mean(coherence_k)

        # coherence_l
        tmp_intra_topic_l = list(combinations(topic_l_words, 2))
        coherence_l = []
        for j in range(len(tmp_intra_topic_l)):
            coherence_tmp = cosine_matrix.loc[topic_l_words_combinations[j][1], topic_l_words_combinations[j][0]]
            coherence_l.append(coherence_tmp)
        coherence_l = np.mean(coherence_l)

        # diversity k_l
        topic_k_l_combinations = list(product(topic_k_words, topic_l_words))
        diversity_k_l_combs = []
        for k in range(len(topic_k_l_combinations)):
            diversity_k_l_tmp = cosine_matrix.loc[topic_k_l_combinations[k][1], topic_k_l_combinations[k][0]]
            diversity_k_l_combs.append(diversity_k_l_tmp)
        diversity_k_l = 1 - np.mean(diversity_k_l_combs)

        quality_k_l = (coherence_k + coherence_l) / 2 * diversity_k_l
        quality_all_tmp.append(quality_k_l)

    if len(quality_all_tmp) > 0:
        quality_all = np.mean(quality_all_tmp)
    else:
        quality_all = 0

    return quality_all