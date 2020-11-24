# -*- coding: utf-8 -*-


'''
------------------------------ Main -----------------------------------
'''

import pandas as pd

import Preparation_CSV as pc        
import feature_importances as fi
import Normalisation_donnees as nor
import Correlation as cor


df_malicious = pd.read_csv("data/l2-malicious.csv")
df_benign = pd.read_csv("data/l2-benign.csv")

df_total_csv = pd.concat([df_malicious,df_benign])

# Preparation des données
df_total_csv_new = pc.Preparation_CSV(df_total_csv)
#df_total_csv_new = df_total_csv.reset_index(drop = True)
# On met notre nouvelle dataframe dans un fichier csv
df_total_csv_new.to_csv("total_csv_copy.csv", sep=';', index=False)

# On normalise les données pour les modèles de ML au cas ou
df_total_csv_normalisee = nor.NormalizeDataset(df_total_csv_new)
df_total_csv_normalisee.to_csv("df_total_csv_normalisee.csv", sep=';', index=False)

df_feature_importances = fi.FeaturesImportances(df_total_csv_normalisee)
df_feature_importances.to_csv("df_feature_importances.csv", sep=';', index=False)

df_correlation = cor.Correlation(df_total_csv_normalisee)
df_correlation.to_csv("df_correlation.csv", sep=';', index=False)