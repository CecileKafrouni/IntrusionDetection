# -*- coding: utf-8 -*-

'''
-------------------- Normalisation des données -------------------------------- 
'''

import numpy as np

def normalize(colonne):
    max_value = colonne.max()
    min_value = colonne.min()
    colonne = np.round((colonne - min_value) / (max_value - min_value), 5)
    return colonne

def NormalizeDataset(dataset) :
    for colonne in dataset.columns:
        if(dataset[colonne].dtypes == 'float64' or dataset[colonne].dtypes == 'int64' and colonne != 'SourcePort' and colonne != 'DestinationPort'):
           dataset[colonne] = normalize(dataset[colonne])
           
    return dataset
