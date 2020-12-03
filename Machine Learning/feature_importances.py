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

def FeaturesImportances(df, colonne):
    #df.astype(np.float32)
    X = df.drop([colonne,'SourceIP','DestinationIP'], axis = 1)
    y = df[colonne]
    X.astype(np.float32)
    
    
    '''ip2float = lambda ip: reduce(lambda a,b: float(a)*256 + float(b), ip.split('.')) 
    
    df['SourceIP'] = df['SourceIP'].apply(ip2float)
    df['DestinationIP'] = df['DestinationIP'].apply(ip2float)
    
    for colonne in X.columns:
        a = np.array((X[colonne]))
        X[colonne] = np.around(a, decimals=5)
    '''    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=2)
    
    rf = RandomForestRegressor(n_estimators = 25)
    rf.fit(X_train, y_train)
    
    importances1 = pd.Series(index=X.columns[:12],
                            data=rf.feature_importances_[:12])
    importances_sorted1 = importances1.sort_values()
    importances_sorted1.plot(kind='barh', color='red', title='1')
    plt.show()
    
    importances2 = pd.Series(index=X.columns[12:24],
                            data=rf.feature_importances_[12:24])
    importances_sorted2 = importances2.sort_values()
    importances_sorted2.plot(kind='barh', color='red', title='2')
    plt.show()
    
    importances3 = pd.Series(index=X.columns[24:36],
                            data=rf.feature_importances_[24:36])
    importances_sorted3 = importances3.sort_values()
    importances_sorted3.plot(kind='barh', color='red', title='3')
    plt.show()

# Pour créer une dataframe avec que les colonnes importantes
    
    feature_importances = pd.DataFrame(rf.feature_importances_, index = X_train.columns, columns=[colonne]).sort_values(colonne,ascending=False)
    df_copy = df.copy()
    
    for i in range(0,len(feature_importances)):
        if feature_importances.iloc[i][colonne] < 0.001:
            print(feature_importances.iloc[i])
            del df_copy[feature_importances.iloc[i].name]
  
    return df_copy