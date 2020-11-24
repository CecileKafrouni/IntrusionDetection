# Librairies needed 

import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score

############## DECISION TREE CLASSIFIER ##################

def DTC(df):
    
    t_debut = time.time()
    
    X = df.drop(['Intrusion'], axis = 1)
    y = df['Intrusion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
       
    # Create Decision Tree classifer object
    DTC = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    
    # Train Decision Tree Classifer
    DTC = DTC.fit(X_train,y_train)
    
    y_pred_DTC = DTC.predict(X_test)
    
    print('Report Decision Tree Classifier \n', classification_report(y_test, y_pred_DTC))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour DTClassifier (en sec): ", t_total)
    
    return DTC
    
def DTC_Prediction(df, DTC):    

    X_new_DTC = df
    # Predict and print the label for the new data point X_new
    new_prediction_DTC = DTC.predict(X_new_DTC)
    print("New prediction knn: {}".format(new_prediction_DTC))
    return new_prediction_DTC

############## RANDOM FOREST CLASSIFIER ##################

def RFC(df):
    
    t_debut = time.time()
    
    X = df.drop(['Intrusion'], axis = 1)
    y = df['Intrusion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Create Random Forest classifer object
    RFC = RandomForestClassifier(criterion="entropy", n_estimators=20, random_state=0)
    
    # Train Random Forest Classifer
    RFC = RFC.fit(X_train,y_train)
    
    y_pred_RFC = RFC.predict(X_test)
    
    print('Report Random Forest Classifier \n', classification_report(y_test, y_pred_RFC))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour RFClassifier (en sec): ", t_total)
    
    return RFC

def RFC_Prediction(df, RFC): 
    
    X_new_RFC = df
    # Predict and print the label for the new data point X_new
    new_prediction_RFC = RFC.predict(X_new_RFC)
    print("New prediction knn: {}".format(new_prediction_RFC))
    return new_prediction_RFC

############### XGBOOST CLASSIFIER ########################

def XGB(df):
    
    t_debut = time.time()
    
    X = df.drop(['Intrusion'], axis = 1)
    y = df['Intrusion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Create XGBoost Classifier model
    XGB = XGBClassifier(objective="binary:logistic")
    
    # Train XGBoost Classifer
    XGB = XGB.fit(X_train,y_train)
    
    y_pred_XGB = XGB.predict(X_test)
    
    print('Report Random Forest Classifier \n', classification_report(y_test, y_pred_XGB))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour DTClassifier (en sec): ", t_total)
    
    return XGB

def XGB_Prediction(df,XGB): 
    
    X_new_XGB = df
    # Predict and print the label for the new data point X_new
    new_prediction_XGB = XGB.predict(X_new_XGB)
    print("New prediction knn: {}".format(new_prediction_XGB))
    return new_prediction_XGB

