# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def Correlation(df, colonne):
    #X = df.drop(['Intrusion','SourceIP','DestinationIP'], axis = 1)
    #y = df['Intrusion']
    #X.astype(np.float32)
    
    plt.figure(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    
    #Correlation with output variable
    cor_target = abs(cor[colonne]) #Selecting highly correlated features
    
    relevant_features = cor_target[cor_target>0.1]
     
    df_copy = pd.DataFrame()
    for nom_colonne in df.columns:
        for index, value in relevant_features.items():
            if(nom_colonne == index):
                df_copy[nom_colonne] = df[nom_colonne]
                
    return df_copy
    
    
 
 
    
    
    