'''
----------------------- MAIN INTERFACE ---------------------------
'''

import Modeles_ML as ml
import Interface as gui
import pickle
import pandas as pd

df = pd.read_csv("DoH/df_total_csv_normalisee_DoH.csv", sep=';')

#Loading ML models for DoH

filename_DTC_DoH = 'DoH/finalized_model_DTC_DoH.sav'
filename_RFC_DoH = 'DoH/finalized_model_RFC_DoH.sav'
filename_XGB_DoH = 'DoH/finalized_model_XGB_DoH.sav'

loaded_model_DTC_DoH = pickle.load(open(filename_DTC_DoH, 'rb'))
loaded_model_RFC_DoH = pickle.load(open(filename_RFC_DoH, 'rb'))
loaded_model_XGB_DoH = pickle.load(open(filename_XGB_DoH, 'rb'))

#Loading DL models for Intrusion

filename_DTC_Intrusion = 'Intrusion/finalized_model_DTC_Intrusion.sav'
filename_RFC_Intrusion = 'Intrusion/finalized_model_RFC_Intrusion.sav'
filename_XGB_Intrusion = 'Intrusion/finalized_model_XGB_Intrusion.sav'

loaded_model_DTC_Intrusion = pickle.load(open(filename_DTC_Intrusion, 'rb'))
loaded_model_RFC_Intrusion = pickle.load(open(filename_RFC_Intrusion, 'rb'))
loaded_model_XGB_Intrusion = pickle.load(open(filename_XGB_Intrusion, 'rb'))

df_test=gui.interface(df,'DoH')

df_test = pd.read_csv("df_new.csv", sep=';')


pred_DTC_DoH = ml.DTC_Prediction(df_test, loaded_model_DTC_DoH)
pred_XGB_DoH = ml.XGB_Prediction(df_test, loaded_model_XGB_DoH)
pred_RFC_DoH = ml.RFC_Prediction(df_test, loaded_model_RFC_DoH)


gui.result(pred_DTC_DoH,pred_RFC_DoH,pred_XGB_DoH)




pred_DTC_Intrusion = ml.DTC_Prediction(df_test, loaded_model_DTC_Intrusion)
pred_RFC_Intrusion = ml.RFC_Prediction(df_test, loaded_model_RFC_Intrusion)
pred_XGB_Intrusion = ml.XGB_Prediction(df_test, loaded_model_XGB_Intrusion)

gui.result_intrusion(pred_DTC_Intrusion,pred_RFC_Intrusion,pred_XGB_Intrusion)

