# cta_nlp.py
# find optimal alpha and delta values for topics

import pandas as pd
import numpy as np

from CTApy.calculate_quality_all import calculate_quality_all

def optimize_clusters(grid_alpha, grid_delta, word_importance, cosine_matrix):
    median_distance = np.median(cosine_matrix)
    interpretability_matrix = np.zeros((len(grid_delta), len(grid_alpha)))
    index_m = 0
    for m in grid_delta:
        delta = m
        index_n = 0
        for n in grid_alpha:
            min_distance = median_distance * n
            words_to_drop = []
            cluster_list_groups = []
            mean_shap = []
            impact_shap = []
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
                    word_importance_clustering = word_importance[word_importance['Variable'].isin(words_to_drop) == False].reset_index(drop=True)
                else:
                    words_to_drop.extend([word_start])
                    word_importance_clustering = word_importance[word_importance['Variable'].isin(words_to_drop) == False].reset_index(drop=True)
            quality_all = calculate_quality_all(cluster_list_groups, cosine_matrix)
            log_nwords = np.log(len([item for sublist in cluster_list_groups for item in sublist]) + 1)
            interpretability = log_nwords * quality_all
            interpretability_matrix[index_m, index_n] = interpretability
            index_n = index_n + 1
        index_m = index_m + 1
    ind = np.unravel_index(np.argmax(interpretability_matrix, axis=None), interpretability_matrix.shape)
    delta = grid_delta[ind[0]]
    alpha = grid_alpha[ind[1]]
    return alpha, delta