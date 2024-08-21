# ctapy.py
# preprocessing

import numpy as np
import pandas as pd
import spacy
import nltk
from gensim.models import fasttext

def preprocessing(data, text_var, outcome_var, metadata_vars, spacy_model, fasttext_model):
    # Extract necessary columns from data
    text = data[text_var]
    outcome = data[outcome_var]
    metadata = data[metadata_vars]

    # Load the spacy model
    nlp = spacy.load(spacy_model)

    # Remove stop words and lemmatize text
    text = text.apply(lambda x: ' '.join([token.text for token in nlp(x) if not token.is_stop]))
    
    # lemmatize text

    text_lemma = text.apply(lambda x: ' '.join([token.lemma_ for token in nlp(x)]))



    # Create a new DataFrame to save
    df_tosave = pd.DataFrame({'text_lemma': text_lemma , 'outcome': outcome, 'text_raw': data[text_var]})
    df_tosave = pd.concat([df_tosave, metadata], axis=1)
    df_tosave = df_tosave[df_tosave['text_lemma'].str.len() > 0].reset_index(drop=True)

    # Prepare data for the fasttext model
    data_words = df_tosave['text_lemma'].str.split().tolist()

    # Load pre-trained model
    pretrained_model = fasttext.load_facebook_model(fasttext_model)

    # Update the model with new data
    pretrained_model.build_vocab(corpus_iterable=data_words, update=True)
    pretrained_model.train(corpus_iterable=data_words, total_examples=len(data_words), epochs=pretrained_model.epochs)
    
    return df_tosave, pretrained_model