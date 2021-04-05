# -*- coding: utf-8 -*-

'''
----------------------- MAIN INTERFACE ---------------------------
'''
import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
import Modeles_ML as ml
import Modeles_ML_autres as ml_bis
import Modeles_DL as dl
import Interface as gui
import pickle
import pandas as pd
from keras.models import load_model
import Preparation_CSV as pc
import numpy as np
import Normalisation_donnees as nor

#df = pd.read_csv("DoH/df_total_csv_normalisee_DoH.csv", sep=';')

df=pd.DataFrame()

#Loading ML models for DoH

filename_DTC_DoH = 'DoH/finalized_model_DTC_DoH.sav'
filename_RFC_DoH = 'DoH/finalized_model_RFC_DoH.sav'
filename_XGB_DoH = 'DoH/finalized_model_XGB_DoH.sav'

filename_GNB_DoH = 'DoH/finalized_model_GNB_DoH.sav'
filename_KNN_DoH = 'DoH/finalized_model_KNN_DoH.sav'
filename_SVM_DoH = 'DoH/finalized_model_SVM_DoH.sav'

loaded_model_DTC_DoH = pickle.load(open(filename_DTC_DoH, 'rb'))
loaded_model_RFC_DoH = pickle.load(open(filename_RFC_DoH, 'rb'))
loaded_model_XGB_DoH = pickle.load(open(filename_XGB_DoH, 'rb'))

#loaded_model_GNB_DoH = pickle.load(open(filename_GNB_DoH, 'rb'))
#loaded_model_KNN_DoH = pickle.load(open(filename_KNN_DoH, 'rb'))

#Loading DL models for Intrusion

#filename_Simple_DL_Model_Intrusion = 'Intrusion/finalized_model_Simple_DL_Model_Intrusion.h5'
#loaded_model_Simple_DL_Model_Intrusion = load_model(filename_Simple_DL_Model_Intrusion)

filename_Conv1D_Model_Intrusion = 'Intrusion/Conv1D.h5'
filename_Conv2D_Model_Intrusion = 'Intrusion/Conv2D.h5'
#filename_LSTM_Model_Intrusion = 'Intrusion/LSTM.h5'

loaded_model_Conv1D_Model_Intrusion = load_model(filename_Conv1D_Model_Intrusion)
loaded_model_Conv2D_Model_Intrusion = load_model(filename_Conv2D_Model_Intrusion)
#loaded_model_LSTM_Model_Intrusion = load_model(filename_LSTM_Model_Intrusion)

#interface de debut
radio_value = gui.beginInterface()

if radio_value == 'Test' : 
    nb = int(np.random.randint(1,11))
    df = pd.read_csv("Tests_interface/df_test_"+str(nb)+".csv", sep=';')


# On appelle l'interface principale
button_value = gui.interface(df)


if(button_value == 'Ok'):

    df_test = pd.read_csv("df_new.csv", sep=';')
    
        
    #on prepare les données insérées par l'utilisateur  
    # enlever valeur NaN, transformer IP en int
    for colonne in df_test.columns:
        if colonne == 'Timestamp' or colonne == 'TimeStamp' or colonne == 'index' or colonne == 'Label' or colonne == 'Unnamed: 0':
            del df_test[colonne]

        if type(df_test[colonne][0]) == str and colonne != 'SourceIP' and colonne != 'DestinationIP':
            print('ok')
            df_test[colonne][0] = 0
    
    
    df_test = pc.IP2Int(df_test, 'SourceIP')
    df_test = pc.IP2Int(df_test, 'DestinationIP')
    df_test = pc.nettoyage(df_test)
    
    # Stocke predictions
    pred_DTC_DoH = ml.DTC_Prediction(df_test, loaded_model_DTC_DoH)
    pred_XGB_DoH = ml.XGB_Prediction(df_test, loaded_model_XGB_DoH)
    pred_RFC_DoH = ml.RFC_Prediction(df_test, loaded_model_RFC_DoH)
    
    #df_test_norm = nor.NormalizeDataset(df_test)
    #pred_GNB_DoH = ml_bis.GNB_Prediction(df_test_norm, loaded_model_GNB_DoH)
    pred_GNB_DoH=0
    #pred_KNN_DoH = ml_bis.KNN_Prediction(df_test_norm, loaded_model_KNN_DoH)
    pred_KNN_DoH = 0
    
    pred_Conv1D_Model_Intrusion = dl.Conv1D_Prediction(df_test, loaded_model_Conv1D_Model_Intrusion)
    pred_Conv2D_Model_Intrusion = dl.Conv2D_Prediction(df_test, loaded_model_Conv2D_Model_Intrusion)
   
    
    # Pop up
    gui.result(pred_DTC_DoH,pred_RFC_DoH, pred_XGB_DoH, 
               pred_GNB_DoH,pred_KNN_DoH, 
               pred_Conv1D_Model_Intrusion,
               pred_Conv2D_Model_Intrusion)
    

