'''
------------------------------ Features Importances -----------------------------------
'''
import ipaddress
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from functools import reduce

def FeaturesImportances(df, colonne, model, model_name):
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    model_FI = model.fit(X,y)
    
    importances1 = pd.Series(index=X.columns[:13],
                            data=model_FI.feature_importances_[:13])
    importances_sorted1 = importances1.sort_values()
    importances_sorted1.plot(kind='barh', color='red', title= model_name +'- Features importance 1/3')
    plt.show()
    
    importances2 = pd.Series(index=X.columns[13:26],
                            data=model_FI.feature_importances_[13:26])
    importances_sorted2 = importances2.sort_values()
    importances_sorted2.plot(kind='barh', color='red', title=model_name +'- Features importance 2/3')
    plt.show()
    
    importances3 = pd.Series(index=X.columns[26:],
                            data=model_FI.feature_importances_[26:])
    importances_sorted3 = importances3.sort_values()
    importances_sorted3.plot(kind='barh', color='red', title=model_name +'- Features importance 3/3')
    plt.show()