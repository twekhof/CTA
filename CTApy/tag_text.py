# cta.py
# tag text

import pandas as pd
def tag_text(text,outcome,cluster_list_groups_pos,cluster_list_groups_neg,shap_values_matrix,name_pos,name_neg,tag_type):
    # Create dummy variables for each sublist in cluster_list_groups_pos
    text_from_model=text
    for i, sublist in enumerate(cluster_list_groups_pos, start=1):
        var_name = f'topic pos {i}'
        globals()[var_name] = text_from_model.apply(lambda x: any(word in x for word in sublist)).astype(int)

    for i, sublist in enumerate(cluster_list_groups_neg, start=1):
        var_name = f'topic neg {i}'
        # Create a dummy variable for each sublist
        globals()[var_name] = text_from_model.apply(lambda x: any(word in x for word in sublist)).astype(int)

    # Now you can access the variables like topic_pos_1, topic_pos_2, etc.

    # Create a DataFrame with the topic dummies and the variable "outcome"
    df_reg = pd.DataFrame()
    for i in range(1, len(cluster_list_groups_pos) + 1):
        df_reg[f'{name_pos} Topic {i}'] = globals()[f'topic pos {i}']
    for i in range(1, len(cluster_list_groups_neg) + 1):
        df_reg[f'{name_neg} Topic {i}'] = globals()[f'topic neg {i}']
    df_reg['outcome'] = pd.to_numeric(outcome)
    # Create a new dataframe with the column names and their means
    df_means = pd.DataFrame({
        'Topic': df_reg.columns[:-1],  # Exclude the 'outcome' column
        'Share [%]': df_reg[df_reg.columns[:-1]].mean()*100  # Calculate the mean of each column, excluding 'outcome'
    })
    df_means.reset_index(drop=True, inplace=True)
    
    print(df_means)
    if tag_type=='indicator':
        return df_reg
    if tag_type=='shap':
        # create a variable for each topic with the total shap value for all words in topic per answer

        for i, sublist in enumerate(cluster_list_groups_pos, start=1):
            var_name = f'topic pos {i}'
            globals()[var_name] = shap_values_matrix[sublist].sum(axis=1)

        for i, sublist in enumerate(cluster_list_groups_neg, start=1):
            var_name = f'topic neg {i}'
            globals()[var_name] = shap_values_matrix[sublist].sum(axis=1)

            # Create a DataFrame with the topic weights and the variable "outcome"
        df_reg = pd.DataFrame()
        for i in range(1, len(cluster_list_groups_pos) + 1):
            df_reg[f'{name_pos} Topic {i}'] = globals()[f'topic pos {i}']
        for i in range(1, len(cluster_list_groups_neg) + 1):
            df_reg[f'{name_neg} Topic {i}'] = globals()[f'topic neg {i}']*(-1)
        #df_reg['outcome'] = outcome
        df_reg['outcome'] = pd.to_numeric(outcome)
        return df_reg