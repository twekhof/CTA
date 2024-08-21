# cta_nlp.py
# allocate topics


import pandas as pd
import numpy as np
import nltk
from scipy.stats import sem, ttest_1samp
from collections import Counter

from CTApy.create_cosine_matrix import create_cosine_matrix
from CTApy.optimize_clusters import optimize_clusters
from CTApy.calculate_clusters import calculate_clusters
from CTApy.calculate_quality_all import calculate_quality_all

DEFAULT_ALPHA_RANGE = np.arange(0.4, 4, 0.2)
DEFAULT_DELTA_RANGE = np.arange(0.4, 1, 0.1)

def allocate_topics(shap_values_matrix, p_value_threshold, method, num_topics, text_from_model, pretrained_model, alpha_range=None, delta_range=None):
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Variable': shap_values_matrix.columns,
        'word_importance': shap_values_matrix.mean(),
        'word_se': sem(shap_values_matrix),
        'word_tval': [ttest_1samp(shap_values_matrix[col], 0)[0] for col in shap_values_matrix.columns],
        'word_pval': [ttest_1samp(shap_values_matrix[col], 0)[1] for col in shap_values_matrix.columns]
    }).fillna(0)

    feature_importance['word_importance_abs'] = abs(feature_importance['word_importance'])

    # Calculate word frequency
    words = [word for sublist in text_from_model.apply(nltk.word_tokenize) for word in sublist]
    word_freq = dict(nltk.FreqDist(words))

    feature_importance['freq'] = [word_freq.get(word, 0) for word in feature_importance['Variable']]
    feature_importance['impact'] = feature_importance['freq'] * feature_importance['word_importance_abs']

    # Filter features based on p-value and importance
    pos_features = feature_importance[(feature_importance['word_pval'] < p_value_threshold) & (feature_importance['word_importance'] > 0)]
    neg_features = feature_importance[(feature_importance['word_pval'] < p_value_threshold) & (feature_importance['word_importance'] < 0)]

    pos_features = pos_features.sort_values('word_importance_abs', ascending=False)
    neg_features = neg_features.sort_values('word_importance_abs', ascending=False)

    # Create cosine matrices
    cosine_matrix_pos = create_cosine_matrix(pretrained_model, pos_features['Variable'])
    cosine_matrix_neg = create_cosine_matrix(pretrained_model, neg_features['Variable'])

    # Set default alpha and delta ranges if not provided
    alpha_range = DEFAULT_ALPHA_RANGE if alpha_range is None else alpha_range
    delta_range = DEFAULT_DELTA_RANGE if delta_range is None else delta_range

    # Optimize clusters and calculate quality
    clusters_pos, clusters_neg = [], []  # Initialize clusters_pos and clusters_neg

    for label, features, cosine_matrix in [('positive', pos_features, cosine_matrix_pos), ('negative', neg_features, cosine_matrix_neg)]:
        alpha, delta = optimize_clusters(alpha_range, delta_range, features, cosine_matrix)
        print(f'{label}\nalpha: {alpha}, delta: {delta}')
        clusters, num_clusters = calculate_clusters(method, num_topics, delta, alpha, cosine_matrix, features)
        quality = calculate_quality_all(clusters, cosine_matrix)

        print(f'{label} topics:\n{clusters}\n')
        print(f'quality {label}:\n{quality}\n')
        print(f'There were a total of {num_clusters} {label} topics.')
        print('')


        if label == 'positive':
            clusters_pos = clusters
        else:
            clusters_neg = clusters

    return clusters_pos, clusters_neg

