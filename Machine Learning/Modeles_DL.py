# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("Intrusion/df_total_csv_normalisee_Intrusion.csv", sep=';')

def DL_simple(df):
    X = df.drop(['Intrusion'], axis = 1)
    y = df['Intrusion']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
    
    # instantiate the model, add hidden and output layers
    model2=Sequential()
    model2.add(Dense(4, input_shape=(33,), activation='tanh'))
    model2.add(Dense(1, activation='sigmoid'))
    
    # 
    # compile and summarize the model
    model2.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model2.summary()
    
    # train the model
    model2.fit(X_train, y_train, epochs=3)
    
    # evaluate the model accuracy on test data
    print('model accuracy on test data: ', np.round(model2.evaluate(X_test, y_test, verbose=0)[1],4))


DL_simple(df)