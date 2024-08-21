# cta.py
# word importance


import shap
import scipy as sp
from tqdm.notebook import tqdm
import nltk
import numpy as np
import pandas as pd
import torch

# Use SHAP to obtain feature importance

def word_importance(model, tokenizer, text_bert, min_len, min_freq):
    # Function to check if the code is running in a Jupyter notebook
    def is_notebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    # Use appropriate tqdm based on the environment
    if is_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Define a prediction function, use CPU
    print('calculating SHAP values...')

    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x])
        attention_mask = (tv != 0).long()
        outputs = model(tv, attention_mask=attention_mask)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:,1]) # use one vs rest logit units
        return val


    # Create an explainer object
    explainer = shap.Explainer(f, tokenizer, silent=True)

    # Initialize a progress bar
    pbar = tqdm(total=len(text_bert))

    # Initialize a list to store SHAP values
    shap_values = []

    # Loop over inputs
    for text in text_bert:
        # Explain the model's predictions on text
        shap_value = explainer(pd.Series(text), fixed_context=1)
        shap_values.append(shap_value)

        # Update the progress bar
        pbar.update(1)


    # Close the progress bar
    pbar.close()

    # Tokenize the text and flatten the list of words
    words = [word for sublist in text_bert.apply(nltk.word_tokenize) for word in sublist]

    # Calculate word frequency
    word_freq = dict(nltk.FreqDist(words))

    # Filter unique words with more than min_len characters and occurring at least min_freq times
    unique_words_filtered = [word for word in set(words) if len(word) >= min_len and word_freq[word] >= min_freq]

    # make a dataframe out of the SHAP values lists

    shap_allwords=pd.DataFrame(np.zeros((text_bert.shape[0], len(unique_words_filtered))))
    shap_allwords.columns=unique_words_filtered


    index=0

    for i in range(text_bert.shape[0]):
        words_tmp = shap_values[i].data
        shap_tmp = shap_values[i].values
        for j in range(len(words_tmp[0])):
            word_tmp = words_tmp[0][j].lstrip()
            if word_tmp in unique_words_filtered:
                shap_allwords.loc[i, word_tmp] = shap_tmp[0][j]


    return shap_allwords