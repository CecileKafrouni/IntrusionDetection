# -*- coding: utf-8 -*-


'''
------------------------------ Main -----------------------------------
'''

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import pandas as pd
import pickle

import Preparation_CSV as pc        
import feature_importances as fi
import Normalisation_donnees as nor
import Correlation as cor
import Modeles_ML as ml
import Split as sp

def Main():
    reponse = input('Bonjour, que voulez vous tester ? \n A. DoH \n B. Intrusion \n\n')
    
    if(reponse == 'A'):
        print('\nVous avez répondu A, c\'est parti pour DoH\n')
        colonne = 'DoH'
        
        df_doh = pd.read_csv("data/l1-doh.csv")
        df_nondoh = pd.read_csv("data/l1-nondoh.csv")
    
        df_total_csv_DoH = pd.concat([df_nondoh,df_doh])
    
        # Preparation des données
        print("Nous nettoyons les données ...")
        df_total_csv_new_DoH = pc.Preparation_CSV(df_total_csv_DoH, colonne)
        df_total_csv_new_DoH.to_csv("DoH/total_csv_copy_DoH.csv", sep=';', index=False)
        
        # On normalise les données pour les modèles de ML au cas ou
        print("Nous normalisons les données ...")
        df_total_csv_normalisee_DoH = nor.NormalizeDataset(df_total_csv_new_DoH)
        df_total_csv_normalisee_DoH.to_csv("DoH/df_total_csv_normalisee_DoH.csv", sep=';', index=False)
        
        # On split les adresses IP
        df_split = sp.splitIp(df_total_csv_normalisee_DoH, 'SourceIP')
        df_split = sp.splitIp(df_split, 'DestinationIP')
        df_split.to_csv("DoH/df_total_csv_ipsplit_DoH.csv", sep=';', index=False)
        
        '''
        # Features importances
        df_feature_importances_DoH = fi.FeaturesImportances(df_total_csv_normalisee_DoH, colonne)
        df_feature_importances_DoH.to_csv("DoH/df_feature_importances_DoH.csv", sep=';', index=False)
        
        # Correlation
        df_correlation_DoH = cor.Correlation(df_total_csv_normalisee_DoH, colonne)
        df_correlation_DoH.to_csv("DoH/df_correlation_DoH.csv", sep=';', index=False)
        '''
        
        # Modele ML
        print("Nous démarrons les modèles de Machine Learning ...")
        # DTC Decision Tree Classifier
        DTC_DoH = ml.DTC(df_split, colonne)
        filename_DTC_DoH = 'DoH/finalized_model_DTC_DoH.sav'
        pickle.dump(DTC_DoH, open(filename_DTC_DoH, 'wb'))
       
        # RFC Random Forest Classifier
        RFC_DoH = ml.RFC(df_total_csv_normalisee_DoH, colonne)
        filename_RFC_DoH = 'DoH/finalized_model_RFC_DoH.sav'
        pickle.dump(RFC_DoH, open(filename_RFC_DoH, 'wb'))
        
        # XGB XGBoost Classifier        
        XGB_DoH = ml.XGB(df_split, colonne)
        filename_XGB_DoH = 'DoH/finalized_model_XGB_DoH.sav'
        pickle.dump(XGB_DoH, open(filename_XGB_DoH, 'wb'))
        
        
    elif(reponse == 'B'):
        print('\nVous avez répondu B, c\'est parti pour Intrusion\n')
        colonne = 'Intrusion'
        
        df_malicious = pd.read_csv("data/l2-malicious.csv")
        df_benign = pd.read_csv("data/l2-benign.csv")
    
        df_total_csv_Intrusion = pd.concat([df_malicious,df_benign])
    
        # Preparation des données
        print("Nous nettoyons les données ...")
        df_total_csv_new_Intrusion = pc.Preparation_CSV(df_total_csv_Intrusion, colonne)
        df_total_csv_new_Intrusion.to_csv("Intrusion/total_csv_copy_Intrusion.csv", sep=';', index=False)
        
        # On normalise les données pour les modèles de ML au cas ou
        print("Nous normalisons les données ...")
        df_total_csv_normalisee_Intrusion = nor.NormalizeDataset(df_total_csv_new_Intrusion)
        df_total_csv_normalisee_Intrusion.to_csv("Intrusion/df_total_csv_normalisee_Intrusion.csv", sep=';', index=False)
        
        # On split les adresses IP
        df_split = sp.splitIp(df_total_csv_normalisee_Intrusion, 'SourceIP')
        df_split = sp.splitIp(df_split, 'DestinationIP')
        df_split.to_csv("Intrusion/df_total_csv_ipsplit_Intrusion.csv", sep=';', index=False)
        
        '''
        # Features importances
        df_feature_importances_Intrusion = fi.FeaturesImportances(df_split, colonne)
        df_feature_importances_Intrusion.to_csv("Intrusion/df_feature_importances_Intrusion.csv", sep=';', index=False)
    
        # Correlation
        df_correlation_Intrusion = cor.Correlation(df_split, colonne)
        df_correlation_Intrusion.to_csv("Intrusion/df_correlation_Intrusion.csv", sep=';', index=False)
        '''
        
        # Modele ML
        print("Nous démarrons les modèles de Machine Learning ... \n\n")
        # DTC Decision Tree Classifier
        DTC_Intrusion = ml.DTC(df_split, colonne)
        filename_DTC_Intrusion = 'Intrusion/finalized_model_DTC_Intrusion.sav'
        pickle.dump(DTC_Intrusion, open(filename_DTC_Intrusion, 'wb'))
       
        # RFC Random Forest Classifier
        RFC_Intrusion = ml.RFC(df_total_csv_normalisee_Intrusion, colonne)
        filename_RFC_Intrusion = 'Intrusion/finalized_model_RFC_Intrusion.sav'
        pickle.dump(RFC_Intrusion, open(filename_RFC_Intrusion, 'wb'))
        
        # XGB XGBoost Classifier        
        XGB_Intrusion = ml.XGB(df_split, colonne)
        filename_XGB_Intrusion = 'Intrusion/finalized_model_XGB_Intrusion.sav'
        pickle.dump(XGB_Intrusion, open(filename_XGB_Intrusion, 'wb'))
        
    else:
        print('\nVous n\' avez pas bien répondu à la question, réessayez svp')
        Main()

# On appelle notre fonction main
Main()