# -*- coding: utf-8 -*-

'''
----------------------- MAIN INTERFACE ---------------------------
'''


import Modeles_ML as ml
import Interface as gui
import pickle
import pandas as pd

df = pd.read_csv("DoH/df_total_csv_normalisee_DoH.csv", sep=';')


filename_DTC_DoH = 'DoH/finalized_model_DTC_DoH.sav'
filename_RFC_DoH = 'DoH/finalized_model_RFC_DoH.sav'
filename_XGB_DoH = 'DoH/finalized_model_XGB_DoH.sav'

loaded_model_DTC_DoH = pickle.load(open(filename_DTC_DoH, 'rb'))
loaded_model_RFC_DoH = pickle.load(open(filename_RFC_DoH, 'rb'))
loaded_model_XGB_DoH = pickle.load(open(filename_XGB_DoH, 'rb'))

df_test=gui.interface(df,'DoH')

df_test = pd.read_csv("df_new.csv", sep=';')
pred_DTC_DoH = ml.DTC_Prediction(df_test, loaded_model_DTC_DoH)
pred_RFC_DoH = ml.RFC_Prediction(df_test, loaded_model_RFC_DoH)
pred_XGB_DoH = ml.XGB_Prediction(df_test, loaded_model_XGB_DoH)

gui.result(pred_DTC_DoH,pred_RFC_DoH,pred_XGB_DoH)




filename_DTC_Intrusion = 'Intrusion/finalized_model_DTC_Intrusion.sav'
filename_RFC_Intrusion = 'Intrusion/finalized_model_RFC_Intrusion.sav'
filename_XGB_Intrusion = 'Intrusion/finalized_model_XGB_Intrusion.sav'