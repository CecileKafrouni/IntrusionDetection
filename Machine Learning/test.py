# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:42:42 2020

@author: etudiant
"""
import pandas as pd
import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def SVM(df):
    
    t_debut = time.time()
    
    X = df.drop(['Intrusion'], axis = 1)
    y = df['Intrusion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Create Support Vector Machine
    SVM = svm(criterion="entropy", n_estimators=20, random_state=0)
    
    # Train Random Forest Classifer
    SVM = svm.fit(X_train,y_train)
    
    y_pred_SVM = SVM.predict(X_test)
    
    print('Report Support Vector Machine \n', classification_report(y_test, y_pred_SVM))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour SVM classifier (en sec): ", t_total)
    
    return SVM
