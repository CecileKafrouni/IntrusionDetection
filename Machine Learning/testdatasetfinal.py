# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:02:13 2021

@author: etudiant
"""

#import feature_importances as fi
import ROC_curve as roc
import pandas as pd
#import feature_importances as fi
#import cross_validation as cv

#Math modules
import numpy as np
import time
from scipy.stats import randint

#Models tools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report

#Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import preprocessing


#lire les données 
data = pd.read_csv("df_total_csv_normalisee_Intrusion.csv",sep=';')
#data = pd.read_csv("df_total_csv_normalisee_DoH.csv",sep=';')

####GNb##############
'''
def GNB(df, colonne):
    print('Compiling the gnb ...\n')
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    GNB = GaussianNB()
       
    # Create k neighborsClassoficer
    GNB.fit(X_train,y_train)
    
    
    y_pred_GNB = GNB.predict(X_test)
    
    
    
    print('Report GNB \n', classification_report(y_test, y_pred_GNB))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour   GNB (en sec): ", t_total)
    
    roc.ROC_curve(df, colonne, GNB, 'GNB')
    
    return GNB
#GNB(data,'Intrusion')
GNB(data,'DoH')




'''


#exemple dtc
'''
def DTC_Randomized_Search(df, colonne):
    print('Compiling the Randomized Search for DTC ...\n')
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    # Setup the parameters and distributions to sample from: param_dist
    param_grid = {"max_depth": [3, None],
                  "max_features": randint(1, 9),
                  "min_samples_leaf": randint(1, 9),
                  "criterion": ["gini", "entropy"]}
    
    # Instantiate a Decision Tree classifier: tree
    DTC = DecisionTreeClassifier()
    
    # Instantiate the RandomizedSearchCV object: tree_cv
    rs_dtc = RandomizedSearchCV(DTC, param_grid)
    
    # Fit it to the data
    rs_dtc.fit(X_train,y_train)
    
    # Print the tuned parameters and score
    print("Les meilleurs parametres sont :\n {}".format(rs_dtc.best_params_))
    print("Le meilleur score est : {} \n".format(np.round(rs_dtc.best_score_,4)))
    
    return rs_dtc
    
    
def DTC(df, colonne):
    
    #rs_dtc = DTC_Randomized_Search(df, colonne)
    
    
    t_debut = time.time()
    print("Training Decision Tree Classifier Algo ...\n ")
    # Create Decision Tree classifer object
    DTC = DecisionTreeClassifier(max_depth=2,criterion='entropy')

    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
       
    
    #DTC.set_params(**rs_dtc.best_params_)
    
    # Train Decision Tree Classifer
    DTC = DTC.fit(X_train,y_train)
    
    
    y_pred_DTC = DTC.predict(X_test)
    
    # Metriques du Decision Tree Classifier
    
    print('\tReport Decision Tree Classifier \n\n', classification_report(y_test, y_pred_DTC))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour DTClassifier (en sec): ", np.round(t_total,4))
    
    
    #fi.FeaturesImportances(df, colonne, DTC, 'Decision Tree Classifier')  
    roc.ROC_curve(df, colonne, DTC, 'Decision Tree Classifier')
    #cv.cross_validation(df, colonne, DTC, 'Decision Tree Classifier')
        
    return DTC
    
def DTC_Prediction(df, DTC):    

    X_new_DTC = df
    # Predict and print the label for the new data point X_new
    new_prediction_DTC = DTC.predict(X_new_DTC)
    print("New prediction DTC: {}".format(new_prediction_DTC))
    return new_prediction_DTC

DTC(data,'Intrusion')
'''
################################# KNN ###################################

def KNN(df, colonne):
    print('Compiling the KNN ...\n')
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    KNN = KNeighborsClassifier(n_neighbors=30)
       
    # Create k neighborsClassoficer
    KNN.fit(X_train,y_train)
    
    
    y_pred_KNN = KNN.predict(X_test)
    
    
    
    print('Report KNN \n', classification_report(y_test, y_pred_KNN))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour   KNNclassifier (en sec): ", t_total)
    
    roc.ROC_curve(df, colonne, KNN, 'KNN')
    
    return KNN



KNN(data,'Intrusion')



################################↕SVM#############################"
'''
def SVM(df, colonne):
    print('Compiling the SVM ...\n')
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
   
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    X_train = preprocessing.scale(X_train)

    X_test = preprocessing.scale(X_test)
    
    # Create Support Vector Machine
    SVM = svm.SVC()
    
    # Train Random Forest Classifer
    SVM.fit(X_train,y_train)
    
    y_pred_SVM = SVM.predict(X_test)
    
    print('Report Support Vector Machine \n', classification_report(y_test, y_pred_SVM))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour SVM classifier (en sec): ", t_total)
    roc.ROC_curve(df, colonne, SVM , 'SVM')
    return SVM

#SVM(data,'Intrusion')
SVM(data,'DoH')
'''



