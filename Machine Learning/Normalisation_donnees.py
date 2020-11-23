# -*- coding: utf-8 -*-

''' -------------------- Normalisation des données -------------------------------- '''

import pandas as pd

def normalize(colonne):
    max_value = colonne.max()
    min_value = colonne.min()
    colonne = (colonne - min_value) / (max_value - min_value)
    return colonne

def NormalizeDataset(dataset) :
    for colonne in dataset.columns:
        if(dataset[colonne].dtypes == 'float64' or dataset[colonne].dtypes == 'int64' ):
           dataset[colonne] = normalize(dataset[colonne])
           
    #dataset_normalise = normalize(dataset.iloc[:,2:len(dataset)])
    #dataset_final_normalise = pd.concat([dataset.iloc[:,0:2], dataset_normalise, dataset.iloc[:,len(dataset):]], axis=1)
    return dataset

