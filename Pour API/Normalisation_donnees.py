# -*- coding: utf-8 -*-

'''
-------------------- Normalisation des données -------------------------------- 
'''

import numpy as np

def normalize(colonne):
    max_value = colonne.max()
    min_value = colonne.min()
    colonne = np.round((colonne - min_value) / (max_value - min_value), 5)
    return colonne, min_value, max_value

def NormalizeDataset(dataset) :
    for colonne in dataset.columns:
        if(dataset[colonne].dtypes == 'float64' or dataset[colonne].dtypes == 'int64' and colonne != 'SourcePort' and colonne != 'DestinationPort'):
           dataset[colonne], _, _ = normalize(dataset[colonne])
           
    return dataset

def NormalizeNewValues(original_dataset, new_dataset):
    
    original_dataset = original_dataset.drop(['DoH'], axis = 1)
    for colonne in original_dataset.columns:
        if( colonne != 'SourcePort' and colonne != 'DestinationPort' and original_dataset[colonne].dtypes == 'float64' or original_dataset[colonne].dtypes == 'int64'):
           
           original_dataset[colonne], min_value, max_value = normalize(original_dataset[colonne])
           
           new_dataset[colonne] = np.round((new_dataset[colonne] - min_value) / (max_value - min_value), 5)
    return new_dataset