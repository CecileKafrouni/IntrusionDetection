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
import Normalisation_donnees as nor
import Correlation as cor
import Modeles_ML as ml
import Modeles_ML_autres as ml_bis
import Modeles_DL as dl

import Remove_Rows as rr

def Main():
    reponse = input('Bonjour, que voulez vous tester ? \n A. DoH \n B. Intrusion \n\n')
    
    if(reponse == 'A'):
        
        print('\nVous avez répondu A, c\'est parti pour DoH\n')
        colonne = 'DoH'
        
        df_doh = pd.read_csv("data/l1-doh.csv")
        df_nondoh = pd.read_csv("data/l1-nondoh.csv")
        
        df_result_doh, df_doh = rr.removeRows(df_doh)
        df_result_nondoh, df_nondoh = rr.removeRows(df_nondoh)
    
        df_result = pd.concat([df_result_doh,df_result_nondoh])
        df_result.reset_index(inplace=True)
        df_result.to_csv('df_result.csv', sep = ';')
    
        df_total_csv_DoH = pd.concat([df_nondoh,df_doh])
        df_total_csv_DoH.reset_index(inplace=True)
        
        # Preparation des données
        print("Nous nettoyons les données ...")
        df_total_csv_new_DoH = pc.Preparation_CSV(df_total_csv_DoH, colonne)
        df_total_csv_new_DoH.to_csv("DoH/total_csv_copy_DoH.csv", sep=';', index=False)

        
        # On normalise les données pour les modèles de ML au cas ou
        print("Nous normalisons les données ...")
        df_total_csv_normalisee_DoH = nor.NormalizeDataset(df_total_csv_new_DoH)
        df_total_csv_normalisee_DoH.to_csv("DoH/df_total_csv_normalisee_DoH.csv", sep=';', index=False)
        
        # Correlation
        df_correlation_DoH = cor.Correlation(df_total_csv_normalisee_DoH, colonne)
        df_correlation_DoH.to_csv("DoH/df_correlation_DoH.csv", sep=';', index=False)
        
        
        df_total_csv_normalisee_DoH = pd.read_csv("DoH/df_total_csv_normalisee_DoH.csv", sep=';')
       
        # Modele ML
        print("Nous démarrons les modèles de Machine Learning ...")
        
        # DTC Decision Tree Classifier
        DTC_DoH = ml.DTC(df_total_csv_normalisee_DoH, colonne)
        filename_DTC_DoH = 'DoH/finalized_model_DTC_DoH.sav'
        pickle.dump(DTC_DoH, open(filename_DTC_DoH, 'wb'))
       
        # RFC Random Forest Classifier
        RFC_DoH = ml.RFC(df_total_csv_normalisee_DoH, colonne)
        filename_RFC_DoH = 'DoH/finalized_model_RFC_DoH.sav'
        pickle.dump(RFC_DoH, open(filename_RFC_DoH, 'wb'))
        
        # XGB XGBoost Classifier        
        XGB_DoH = ml.XGB(df_total_csv_normalisee_DoH, colonne)
        filename_XGB_DoH = 'DoH/finalized_model_XGB_DoH.sav'
        pickle.dump(XGB_DoH, open(filename_XGB_DoH, 'wb'))   
        
        # Modeles ML autres
         # GNB Gaussian Naives Bayes
        GNB_DoH = ml_bis.GNB(df_total_csv_normalisee_DoH, colonne)
        filename_GNB_DoH = 'DoH/finalized_model_GNB_DoH.sav'
        pickle.dump(GNB_DoH, open(filename_GNB_DoH, 'wb'))
        
        # KNN KNeighbors Classifier
        KNN_DoH = ml_bis.KNN(df_total_csv_normalisee_DoH, colonne)
        filename_KNN_DoH = 'DoH/finalized_model_KNN_DoH.sav'
        pickle.dump(KNN_DoH, open(filename_KNN_DoH, 'wb'))
        
        # Perceptron 
        Per_DoH = ml_bis.Perceptron(df_total_csv_normalisee_DoH, colonne)
        filename_Per_DoH = 'DoH/finalized_model_Per_DoH.sav'
        pickle.dump(Per_DoH, open(filename_Per_DoH, 'wb'))
        
    
    elif(reponse == 'B'):
        print('\nVous avez répondu B, c\'est parti pour Intrusion\n')
        colonne = 'Intrusion'
        
        df_malicious = pd.read_csv("data/l2-malicious.csv")
        df_benign = pd.read_csv("data/l2-benign.csv")
    
        df_total_csv_Intrusion = pd.concat([df_malicious,df_benign])
        df_total_csv_Intrusion.reset_index(inplace=True)
    
        # Preparation des données
        print("Nous nettoyons les données ...")
        df_total_csv_new_Intrusion = pc.Preparation_CSV(df_total_csv_Intrusion, colonne)
        df_total_csv_new_Intrusion.to_csv("Intrusion/total_csv_copy_Intrusion.csv", sep=';', index=False)
        print('Number of values in Intrusion feature: \n',df_total_csv_new_Intrusion[colonne].value_counts())
        
        # On normalise les données pour les modèles de ML au cas ou
        print("Nous normalisons les données ...")
        df_total_csv_normalisee_Intrusion = nor.NormalizeDataset(df_total_csv_new_Intrusion)
        df_total_csv_normalisee_Intrusion.to_csv("Intrusion/df_total_csv_normalisee_Intrusion.csv", sep=';', index=False)
        
        
        # Correlation
        df_correlation_Intrusion = cor.Correlation(df_total_csv_normalisee_Intrusion, colonne)
        df_correlation_Intrusion.to_csv("Intrusion/df_correlation_Intrusion.csv", sep=';', index=False)
        
        # Modeles DL
        #Simple_DL_Model_Intrusion = dl.DTC(df_total_csv_normalisee_Intrusion, 'Intrusion')
        df_total_csv_normalisee_Intrusion = pd.read_csv('Intrusion/df_total_csv_normalisee_Intrusion.csv', sep=';')
        cnn1D_model = dl.model_cnn_1D(df_total_csv_normalisee_Intrusion, 'Intrusion',nb_layers=2, first_layer_nb_filters=32, layer_nb_filters=16, dropout_alpha=0.5, 
             filter_size=3, nb_epochs=8, batch_size=64)
        cnn1D_model.save('Intrusion/Conv1D.h5')
        
        cnn2D_model = dl.model_cnn_2D(df_total_csv_normalisee_Intrusion, 'Intrusion', nb_layers=1, first_layer_nb_filters=32, layer_nb_filters=64, nb_epochs=8, batch_size=256)
        cnn2D_model.save('Intrusion/Conv2D.h5')
        
        
    else:
        print('\nVous n\' avez pas bien répondu à la question, réessayez svp')
        Main()

# On appelle notre fonction main
Main()

