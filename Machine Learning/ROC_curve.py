# -*- coding: utf-8 -*-

'''
------------------------------ ROC Curve -----------------------------------
'''

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

def ROC_curve(df, colonne, model, model_name):
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.title('Receiver Operating Characteristic for ' + model_name)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()