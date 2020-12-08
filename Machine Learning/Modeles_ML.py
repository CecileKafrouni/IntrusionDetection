# Librairies needed 

import pandas as pd
import time
from scipy.stats import randint

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score

################################# DECISION TREE CLASSIFIER ####################################

def DTC_Randomized_Search(df, colonne):
    print('Compiling the Randomized Search for DTC')
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {"max_depth": [3, None],
                  "max_features": randint(1, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    
    # Instantiate a Decision Tree classifier: tree
    DTC = DecisionTreeClassifier()
    
    # Instantiate the RandomizedSearchCV object: tree_cv
    rs_dtc = RandomizedSearchCV(DTC, param_dist, cv=5)
    
    # Fit it to the data
    rs_dtc.fit(X_train,y_train)
    
    # Print the tuned parameters and score
    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    print("Best score is {}".format(tree_cv.best_score_))
    
    return rs_dtc
    
    
def DTC(df, colonne):
    
    print("Training XGBoost Classifier Algo ... ")
    
    tree_cv = DTC_Randomized_Search(df, colonne)
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
       
    # Create Decision Tree classifer object
    DTC = DecisionTreeClassifier()
    DTC.set_params(**tree_cv.best_params_)
    
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

################################# RANDOM FOREST CLASSIFIER ###################################

def RFC_Randomized_Search(df, colonne):
    
    print('Compiling the Randomized Search for RFC')
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
    # Create a based model
    RFC = RandomForestClassifier()
    # Instantiate the grid search model
    rs_rfc = RandomizedSearchCV(RFC, param_grid, cv = 3)
    rs_rfc.fit(X_train,y_train)
    
    return rs_rfc

def RFC(df, colonne):
    
    rs_rfc = RFC_Randomized_Search(df, colonne)
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Create Random Forest classifer object
    RFC = RandomForestClassifier()
    RFC.set_params(**rs_rfc.best_params_)
    
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

#################################### XGBOOST CLASSIFIER ###########################################
    
def XGBoost_Randomized_Search(df, colonne):
    print('Compiling the Randomized Search for XGBoost')
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    XGB = xgb.XGBClassifier()
    
    param_grid = {
            'silent': [False],
            'max_depth': [6, 10, 15, 20],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
            'gamma': [0, 0.25, 0.5, 1.0],
            'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
            'n_estimators': [100]}

    
    rs_XGB = RandomizedSearchCV(XGB, param_grid, n_iter=20,
                                n_jobs=1, verbose=2, cv=2,
                                scoring='neg_log_loss', refit=False, random_state=42)
    rs_XGB.fit(X_train, y_train)
    
    return rs_XGB
    
def XGB(df, colonne):
    
    rs_XGB = XGBoost_Randomized_Search(df, colonne)
    
    t_debut = time.time()
    print("Training XGBoost Classifier Algo ... ")
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Create XGBoost Classifier model
    XGB = xgb.XGBClassifier()
    XGB.set_params(**rs_XGB.best_params_)
    
    # Train XGBoost Classifer
    XGB = XGB.fit(X_train,y_train)
    
    y_pred_XGB = XGB.predict(X_test)
    
    print('Report XGBoost Classifier \n', classification_report(y_test, y_pred_XGB))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour XGBoost (en sec): ", t_total)
    
    return XGB

def XGB_Prediction(df,XGB): 
    
    X_new_XGB = df
    # Predict and print the label for the new data point X_new
    new_prediction_XGB = XGB.predict(X_new_XGB)
    print("New prediction knn: {}".format(new_prediction_XGB))
    return new_prediction_XGB