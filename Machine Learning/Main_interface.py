# -*- coding: utf-8 -*-

'''
----------------------- MAIN INTERFACE ---------------------------
'''
import sys

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

df = pd.read_csv("DoH/df_total_csv_normalisee_DoH.csv", sep=';')

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

loaded_model_GNB_DoH = pickle.load(open(filename_GNB_DoH, 'rb'))
loaded_model_KNN_DoH = pickle.load(open(filename_KNN_DoH, 'rb'))
loaded_model_SVM_DoH = pickle.load(open(filename_SVM_DoH, 'rb'))

#Loading DL models for Intrusion

filename_Simple_DL_Model_Intrusion = 'Intrusion/finalized_model_Simple_DL_Model_Intrusion.h5'

loaded_model_Simple_DL_Model_Intrusion = load_model(filename_Simple_DL_Model_Intrusion)

# On appelle l'interface principale
button_value = gui.interface(df,'DoH')

if(button_value == 'Ok'):

    df_test = pd.read_csv("df_new.csv", sep=';')
    
    # Stocke predictions
    pred_DTC_DoH = ml.DTC_Prediction(df_test, loaded_model_DTC_DoH)
    pred_XGB_DoH = ml.XGB_Prediction(df_test, loaded_model_XGB_DoH)
    pred_RFC_DoH = ml.RFC_Prediction(df_test, loaded_model_RFC_DoH)
    
    pred_GNB_DoH = ml_bis.GNB_Prediction(df_test, loaded_model_GNB_DoH)
    pred_KNN_DoH = ml_bis.KNN_Prediction(df_test, loaded_model_KNN_DoH)
    pred_SVM_DoH = ml_bis.SVM_Prediction(df_test, loaded_model_SVM_DoH)
    
    pred_Simple_DL_Model_Intrusion = dl.DL_simple_Prediction(df_test, loaded_model_Simple_DL_Model_Intrusion)
    
    # Pop up
    gui.result(pred_DTC_DoH,pred_RFC_DoH,pred_XGB_DoH,pred_Simple_DL_Model_Intrusion, pred_GNB_DoH,pred_KNN_DoH, pred_SVM_DoH)