# -*- coding: utf-8 -*-


'''
------------------------------ Main -----------------------------------
'''

import pandas as pd

import Preparation_CSV as pc        
#import feature_importances_ as fi
import Normalisation_donnees as nor

df_malicious = pd.read_csv("l2-malicious.csv")
df_benign = pd.read_csv("l2-benign.csv")

df_total_csv = pd.concat([df_malicious,df_benign])
df_total_csv = df_total_csv.reset_index(drop = True)

# Preparation des données
df_total_csv_new = pc.Preparation_CSV(df_total_csv)

# On met notre nouvelle dataframe dans un fichier csv
df_total_csv_new.to_csv("total_csv_copy.csv", sep=';', index=False)

# On normalise les données pour les modèles de ML au cas ou
df_total_csv_normalisee = nor.NormalizeDataset(df_total_csv_new)

df_total_csv_normalisee.to_csv("df_total_csv_normalisee.csv", sep=';', index=False)

