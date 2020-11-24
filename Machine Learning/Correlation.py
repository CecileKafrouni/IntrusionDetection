# -*- coding: utf-8 -*-


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def Correlation(df):
    #X = df.drop(['Intrusion','SourceIP','DestinationIP'], axis = 1)
    #y = df['Intrusion']
    #X.astype(np.float32)
    
    plt.figure(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    #Correlation with output variable
    cor_target = abs(cor["Intrusion"]) #Selecting highly correlated features
    
    relevant_features = cor_target[cor_target>0.1]
     
    df_copy = pd.DataFrame()
    for colonne in df.columns:
        for index, value in relevant_features.items():
            if(colonne == index):
                df_copy[colonne] = df[colonne]
                
    return df_copy
    
    
 
 
    
    
    