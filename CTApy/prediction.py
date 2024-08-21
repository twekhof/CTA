# cta.py
# prediction


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import random
import pandas as pd


def prediction(text_bert,outcome_bert,seed,bert_model,epochs):
    # Predict outcome with BERT
 
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


    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    set_seed(seed) 

    tokenizer = DistilBertTokenizer.from_pretrained(bert_model)
    model = DistilBertForSequenceClassification.from_pretrained(bert_model, num_labels=2)

    texts = text_bert.tolist()
    labels = pd.to_numeric(outcome_bert).tolist()

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    labels = torch.tensor(labels)

    # Create a DataLoader
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    dataloader = DataLoader(dataset, batch_size=16)

    # Choose an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    all_labels = []
    all_predictions = []

    print('This message points out that there are no pre-trained weights for the additional layer from the binary prediction. This is however expected because we train this model now.')

    print('Train model over', epochs, 'epochs:')
    # Initialize a progress bar
    pbar = tqdm(total=epochs)

    # Fine-tuning loop
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()

            optimizer.step()

            # Get predictions for each batch
            preds = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(preds.tolist())
            all_labels.extend(labels.tolist())
        # Update the progress bar
        pbar.update(1)
    # Close the progress bar
    pbar.close()
    return model,tokenizer