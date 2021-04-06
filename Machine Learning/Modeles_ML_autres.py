# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 12:02:13 2021

@author: etudiant
"""

#import feature_importances as fi
import ROC_curve as roc
import feature_importances as fi
import cross_validation as cv

#Math modules
import time

#Models tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score, recall_score

#Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import preprocessing

####GNb##############

def GNB(df, colonne):
    print('Compiling the gnb ...\n')
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_sample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
       
    GNB = GaussianNB(var_smoothing=1e-12)
       
    # Create k neighborsClassoficer
    GNB.fit(X_train,y_train)
    
    
    y_pred_GNB = GNB.predict(X_test)
    
    print('Report GNB \n', classification_report(y_test, y_pred_GNB))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour GNB (en sec): ", t_total)

    
    #fi.FeaturesImportances(df, colonne, XGB, 'XGBoost Classifier')  
    roc.ROC_curve(df, colonne, KNN, 'XGBoost Classifier')
    #cv.cross_validation(df, colonne, XGB, 'XGBoost Classifier')
    
    print("\nMatrice de confusion : \n",confusion_matrix(y_test, y_pred_GNB))
    print("\nAccuracy : ",accuracy_score(y_test, y_pred_GNB))
    print("\nPrecision :", precision_score(y_test, y_pred_GNB))
    print("\nF1 score :", f1_score(y_test, y_pred_GNB))
    print("\nRecall :", recall_score(y_test, y_pred_GNB))
    
    return GNB


    
def GNB_Prediction(df, GNB):    

    X_new_GNB = df
    # Predict and print the label for the new data point X_new
    new_prediction_GNB = GNB.predict(X_new_GNB)
    print("New prediction GNB: {}".format(new_prediction_GNB))
    
    return new_prediction_GNB



################################# KNN ###################################

def KNN(df, colonne):
    print('Compiling the KNN ...\n')
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_sample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
       
    KNN = KNeighborsClassifier(n_neighbors=10)
       
    # Create k neighborsClassoficer
    KNN.fit(X_train,y_train)
    
    
    y_pred_KNN = KNN.predict(X_test)
    
    print('Report KNN \n', classification_report(y_test, y_pred_KNN))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour KNN classifier (en sec): ", t_total)
    
    #fi.FeaturesImportances(df, colonne, XGB, 'XGBoost Classifier')  
    roc.ROC_curve(df, colonne, KNN, 'XGBoost Classifier')
    #cv.cross_validation(df, colonne, XGB, 'XGBoost Classifier')
    
    print("\nMatrice de confusion : \n",confusion_matrix(y_test, y_pred_KNN))
    print("\nAccuracy : ",accuracy_score(y_test, y_pred_KNN))
    print("\nPrecision :", precision_score(y_test, y_pred_KNN))
    print("\nF1 score :", f1_score(y_test, y_pred_KNN))
    print("\nRecall :", recall_score(y_test, y_pred_KNN))
    
    return KNN


 
def KNN_Prediction(df, KNN):    
    X_new_KNN = df
    # Predict and print the label for the new data point X_new
    new_prediction_KNN = KNN.predict(X_new_KNN)
    print("New prediction KNN: {}".format(new_prediction_KNN))
    return new_prediction_KNN

################################↕SVM#############################"

def SVM(df, colonne):
    print('Compiling the SVM ...\n')
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_sample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
       
    X_train = preprocessing.scale(X_train)

    X_test = preprocessing.scale(X_test)
    
    # Create Support Vector Machine
    SVM = svm.SVC(probability=True)
    
    # Train Random Forest Classifer
    SVM.fit(X_train,y_train)
    
    y_pred_SVM = SVM.predict(X_test)
    
    print('Report Support Vector Machine \n', classification_report(y_test, y_pred_SVM))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour SVM classifier (en sec): ", t_total)
    
    #fi.FeaturesImportances(df, colonne, SVM, 'Support Vector Machine')  
    roc.ROC_curve(df, colonne, SVM, 'Support Vector Machine')
    #cv.cross_validation(df, colonne, SVM, 'Support Vector Machine')
    
    return SVM


 
def SVM_Prediction(df, SVM):    

    X_new_SVM = df
    # Predict and print the label for the new data point X_new
    new_prediction_SVM = SVM.predict(X_new_SVM)
    print("New prediction SVM: {}".format(new_prediction_SVM))
    
    return new_prediction_SVM

############################### Perceptron #######################
    
####GNb##############

def Perceptron(df, colonne):
    print('Compiling the Perceptron ...\n')
    
    t_debut = time.time()
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_sample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
       
    Per =Perceptron(class_weight = 'balanced')
       
    Per.fit(X_train,y_train)
    
    
    y_pred_Per = Per.predict(X_test)
    
    print('Report GNB \n', classification_report(y_test, y_pred_Per))
    
    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("Temps pour GNB (en sec): ", t_total)

    
    roc.ROC_curve(df, colonne, KNN, 'XGBoost Classifier')
    
    print("\nMatrice de confusion : \n",confusion_matrix(y_test, y_pred_Per))
    print("\nAccuracy : ",accuracy_score(y_test, y_pred_Per))
    print("\nPrecision :", precision_score(y_test, y_pred_Per))
    print("\nF1 score :", f1_score(y_test, y_pred_Per))
    print("\nRecall :", recall_score(y_test, y_pred_Per))
    
    return Per


    
def Perceptron_Prediction(df, Per):    

    X_new_Per = df
    # Predict and print the label for the new data point X_new
    new_prediction_Per = Per.predict(X_new_Per)
    print("New prediction GNB: {}".format(new_prediction_Per))
    
    return new_prediction_Per