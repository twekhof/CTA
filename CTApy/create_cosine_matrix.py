# cta.py
# define cosine matrix

import pandas as pd
import numpy as np



def create_cosine_matrix(pretrained_model, word_list):
                         
    def similarity_cosine(vec1, vec2):
        cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_similarity
                         
    vocab_index = pd.DataFrame({'feature_names':word_list,'feature_index':range(1,len(word_list)+1)})
    all_words_dict = {a: x-1 for a, x in vocab_index.values}
    word_index = all_words_dict
    EMBED_DIM = 300
    words_not_found = []
    nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words, EMBED_DIM))
    emb_name = []
    index = 0
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        try:
            embedding_vector = pretrained_model.wv[word]
        except KeyError:
            embedding_vector = None
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embedding_matrix[index] = embedding_vector
            emb_name.append(word)
            index = index + 1
        else:
            words_not_found.append(word)
    if words_not_found != []:
        print(f"Null word embeddings: {np.sum(np.sum(embedding_matrix, axis=1) == 0)}")
        print(f"Some of the words not found:\n{' '.join([random.choice(words_not_found) for x in range(0,10)])}")
    n_words_notfound = embedding_matrix.shape[0] - len(emb_name)
    embedding_matrix = np.delete(embedding_matrix, range(embedding_matrix.shape[0] - len(words_not_found), embedding_matrix.shape[0]), 0)
    embedding_matrix_names = pd.DataFrame(np.transpose(embedding_matrix))
    embedding_matrix_names.columns = emb_name
    cosine_matrix = pd.DataFrame(np.zeros((len(emb_name), len(emb_name))), columns=emb_name, index=emb_name)
    for i in range(cosine_matrix.shape[1]):
        word_1 = cosine_matrix.columns[i]
        for j in range(cosine_matrix.shape[1]):
            word_2 = cosine_matrix.index[j]
            cosine_matrix.iloc[j, i] = similarity_cosine(embedding_matrix_names[word_1], embedding_matrix_names[word_2])
    return cosine_matrix