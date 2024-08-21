# ctanlp.py
# find topics

import pandas as pd
import numpy as np


def calculate_clusters(method, n_clusters, delta, alpha, cosine_matrix, word_importance):
    words_to_drop = []
    cluster_list_groups = []
    mean_shap = []
    impact_shap = []
    mean_dist_words_topics = []

    median_distance = np.median(cosine_matrix)
    min_distance = alpha * median_distance
    word_importance_clustering = word_importance[word_importance['Variable'].isin(words_to_drop) == False].reset_index(drop=True)

    while len(word_importance_clustering) > 1:
        word_start = word_importance_clustering['Variable'].loc[0]

        cosine_matrix_tmp = cosine_matrix.copy()
        cosine_matrix_tmp.drop(words_to_drop, inplace=True, axis=0)
        cosine_matrix_tmp.drop(words_to_drop, inplace=True, axis=1)

        word_closest = cosine_matrix_tmp[word_start].nlargest(2).index[1]
        dist_init = cosine_matrix_tmp[word_start].nlargest(2).iloc[1]
        if dist_init > min_distance:
            mean_dist_words = (cosine_matrix_tmp[word_start] + cosine_matrix_tmp[word_closest]) / 2

            dist_threshold = dist_init * delta

            cluster_list_tmp = [word_start, word_closest]

            dist_words = 1
            while (dist_words > dist_threshold) & (len(cosine_matrix_tmp) > len(cluster_list_tmp)):
                mean_dist_words = cosine_matrix_tmp[cluster_list_tmp].mean(axis=1)
                mean_dist_words.drop(cluster_list_tmp, inplace=True)
                word_add = mean_dist_words.nlargest(2).index[0]
                dist_words = mean_dist_words.nlargest(2).iloc[0]
                cluster_list_tmp.append(word_add)
            words_to_drop.extend(cluster_list_tmp)
            cluster_list_groups.append(cluster_list_tmp)
            mean_dist_words_topics_tmp=np.mean(cosine_matrix_tmp[cluster_list_tmp])
            if method != 'freq_only':
                mean_dist_words_topics.append(mean_dist_words_topics_tmp)
                mean_shap.append(word_importance[word_importance['Variable'].isin(cluster_list_tmp)]['word_importance_abs'].mean())
                impact_shap.append(word_importance[word_importance['Variable'].isin(cluster_list_tmp)]['impact'].mean())
            word_importance_clustering = word_importance[word_importance['Variable'].isin(words_to_drop) == False].reset_index(drop=True)
        else:
            words_to_drop.extend([word_start])
            word_importance_clustering = word_importance[word_importance['Variable'].isin(words_to_drop) == False].reset_index(drop=True)

    n_cluster_index = len(cluster_list_groups)
    if n_cluster_index < n_clusters:
        n_clusters_to_use = n_cluster_index
    else:
        n_clusters_to_use = n_clusters

    if method == 'freq_only':
        index_clusters = range(n_clusters_to_use)
    elif method == 'top_shap':
        index_clusters = range(n_clusters_to_use)
    elif method == 'mean_shap':
        index_clusters = np.argpartition(mean_shap, -n_clusters_to_use)[-n_clusters_to_use:]
    elif method == 'impact':
        index_clusters = np.argpartition(impact_shap, -n_clusters_to_use)[-n_clusters_to_use:]
    elif method == 'coherence':
        index_clusters = np.argpartition(mean_dist_words_topics, -n_clusters_to_use)[-n_clusters_to_use:]

    cluster_list_groups = [cluster_list_groups[l] for l in index_clusters]


    return cluster_list_groups, n_cluster_index 