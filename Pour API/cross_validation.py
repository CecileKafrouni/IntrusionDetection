# -*- coding: utf-8 -*-

'''
------------------------------ Cross validation -----------------------------------
'''

from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

def cross_validation(df, colonne, model, model_name):
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_sample(X, y)
    
    scores = cross_val_score(model, X_smote, y_smote, cv=5)

    print("\nAccuracy for " + model_name + ": %0.2f (+/- %0.2f)\n\n" % (scores.mean(), scores.std() * 2))