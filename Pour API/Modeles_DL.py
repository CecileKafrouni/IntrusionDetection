# -*- coding: utf-8 -*-

'''
------------------------------ Modeles de Deep Learning -----------------------------------
'''


# Basic Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot
import time

#Sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import cross_val_score 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, f1_score, recall_score

#Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import EarlyStopping

from keras.layers import LeakyReLU


df=pd.read_csv("Intrusion/df_total_csv_normalisee_Intrusion.csv", sep=';')

X = df.drop(['Intrusion'], axis = 1)
y = df['Intrusion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
def DL_simple(df, colonne):
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
    
    # instantiate the model, add hidden and output layers
    model=Sequential()
    model.add(Dense(4, input_shape=(33,), activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    
    
    # compile and summarize the model
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    
    # train the model
    model.fit(X_train, y_train, epochs=3)
    
    # evaluate the model accuracy on test data
    print('model accuracy on test data: ', np.round(model.evaluate(X_test, y_test, verbose=0)[1],4))
    
    model.save('Intrusion/finalized_model_Simple_DL_Model_Intrusion.h5')
    
    return model

def DL_simple_Prediction(df, DL_simple): 
    
    X_new_DL_simple = df
    new_prediction_DL_simple = DL_simple.predict(X_new_DL_simple)
    new_prediction_DL_simple = np.round(new_prediction_DL_simple)
    
    print("New prediction DLsimple model: {}".format(new_prediction_DL_simple))
    return new_prediction_DL_simple


def create_model(first_layer_activation, hidden_layer_activation, neurons, number_of_layers):
     
    model_opt = Sequential()
    if first_layer_activation=='LeakyReLU':
        model_opt.add(Dense(neurons, input_shape=(33,), activation=LeakyReLU()))
    else:
        model_opt.add(Dense(neurons, input_shape=(33,), activation=first_layer_activation))
    
    for i in range(number_of_layers):
        if hidden_layer_activation=='LeakyReLU':
            model_opt.add(Dense(neurons, activation=LeakyReLU()))
        else:
            model_opt.add(Dense(neurons, activation=hidden_layer_activation))
        
    model_opt.add(Dense(1, activation='sigmoid'))
    model_opt.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model_opt


skmodel = KerasClassifier(build_fn=create_model) 
params = {'first_layer_activation':['relu', 'tanh', 'LeakyReLU'],
          'hidden_layer_activation':['relu', 'tanh', 'LeakyReLU'],
          'batch_size':[8, 46, 64], 
          'epochs':[5, 10, 20, 30],
          'neurons':[32,64,128],
          'number_of_layers':[2,4,6]}

#random_search = RandomizedSearchCV(skmodel, param_distributions=params, cv=3, n_jobs=-1)
#random_search_results = random_search.fit(X_train, y_train, verbose=0) 
#print("Best: {} using {}".format(np.round(random_search_results.best_score_,4), random_search_results.best_params_))

def DL_optimized(df):
    
    # instantiate the model, add hidden and output layers
    model_opt = Sequential()
    model_opt.add(Dense(64, input_shape=(33,), activation='tanh'))
    
    model_opt.add(Dense(64, activation='relu'))
    model_opt.add(Dense(64, activation='relu'))
    model_opt.add(Dense(64, activation='relu'))
    model_opt.add(Dense(64, activation='relu'))
    
    model_opt.add(Dense(1, activation='sigmoid'))
    
    # compile and summarize the model
    model_opt.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model_opt.summary()
    
    # train the model
    model_opt.fit(X_train, y_train, epochs=3, batch_size= 46)
    
    # evaluate the model accuracy on test data
    print('model accuracy on test data: ', np.round(model_opt.evaluate(X_test, y_test, verbose=0)[1],4))
    
#DL_optimized(df)
    


''' DEFINITION MODELES DL '''

def model_cnn_1D(df, colonne, nb_layers, first_layer_nb_filters, layer_nb_filters, dropout_alpha, filter_size, nb_epochs, batch_size):
    
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_sample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=1)

    X_train = X_train.values.reshape(X_train.shape[0],X_train.shape[1],1)

    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
     
    t_debut = time.time()

    model = Sequential()

    model.add(Conv1D(filters=first_layer_nb_filters, kernel_size=filter_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]))) 
    if nb_layers != 0:
      for i in range(0,nb_layers):
        model.add(Conv1D(filters=layer_nb_filters, kernel_size=filter_size, activation='relu'))

    model.add(Dropout(dropout_alpha))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train,y_train, batch_size=batch_size, epochs=nb_epochs,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.0001)])  

    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("\nTemps (en sec): ", np.round(t_total,4))
    

    print('model accuracy on test data: ', np.round(model.evaluate(X_test, y_test, verbose=0)[1],4))

    y_pred = np.round(model.predict(X_test))

    print("\nMatrice de confusion : \n",confusion_matrix(y_test, y_pred))
    print("\nAccuracy : ",accuracy_score(y_test, y_pred))
    print("\nPrecision :", precision_score(y_test, y_pred))
    print("\nF1 score :", f1_score(y_test, y_pred))
    print("\nRecall score :", recall_score(y_test, y_pred))

    # plot accuracy
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model train vs validation accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='lower right')
    pyplot.show()

    # plot loss
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    
    return model


def model_cnn_2D(df, colonne, nb_layers, first_layer_nb_filters, layer_nb_filters, nb_epochs, batch_size): 
  
    X = df.drop([colonne], axis = 1)
    y = df[colonne]
    
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_sample(X, y)
    
    scaler = MinMaxScaler(feature_range=(0, 255))

    scaler = scaler.fit(X_smote)
    X_scaled = scaler.transform(X_smote)
    
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=1)
    X_train = X_train.values.reshape(X_train.shape[0],X_train.shape[1], 1, 1)

    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1, 1)

    t_debut = time.time()

    model = Sequential()
    model.add(Conv2D(first_layer_nb_filters, (3, 3), activation='relu', input_shape=(33, 1, 1), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    if nb_layers != 0:
      for i in range(0,nb_layers):
        model.add(Conv2D(layer_nb_filters, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        print('couche ajoutee')

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    history = model.fit(X_train,y_train, batch_size=batch_size, epochs=nb_epochs,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, min_delta=0.0001)])  

    t_fin = time.time()
    
    t_total = t_fin - t_debut
    
    print("\nTemps (en sec): ", np.round(t_total,4))
    
    print('model accuracy on test data: ', np.round(model.evaluate(X_test, y_test, verbose=0)[1],4))

    y_pred = np.round(model.predict(X_test))

    print("\nMatrice de confusion : \n",confusion_matrix(y_test, y_pred))
    print("\nAccuracy : ",accuracy_score(y_test, y_pred))
    print("\nPrecision :", precision_score(y_test, y_pred))
    print("\nF1 score :", f1_score(y_test, y_pred))
    print("\nRecall score :", recall_score(y_test, y_pred))

    # plot accuracy
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model train vs validation accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='lower right')
    pyplot.show()

    # plot loss
    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
    
    return model

'''
------------------ Predictions pour Conv1D, Conv2D, LSTM ---------------------
'''

def Conv1D_Prediction(df, model): 
    
    X_new = df
    X_new = X_new.values.reshape(X_new.shape[0],X_new.shape[1],1)

    new_prediction_Conv1D = model.predict(X_new)
    new_prediction_Conv1D = np.round(new_prediction_Conv1D)
    
    print("New prediction Conv1D model: {}".format(new_prediction_Conv1D))
    return new_prediction_Conv1D

def Conv2D_Prediction(df, model): 
    
    X_new = df
    X_new = X_new.values.reshape(X_new.shape[0],X_new.shape[1],1,1)

    new_prediction_Conv2D = model.predict(X_new)
    new_prediction_Conv2D = np.round(new_prediction_Conv2D)
    
    print("New prediction Conv2D model: {}".format(new_prediction_Conv2D))
    return new_prediction_Conv2D

# on ne fait plus LSTM car trop long et avec peu d'epoch sous-apprend
def LSTM_Prediction(df, model): 
    
    X_new = df
    new_prediction_LSTM = DL_simple.predict(X_new)
    new_prediction_LSTM = np.round(new_prediction_LSTM)
    
    print("New prediction LSTM model: {}".format(new_prediction_LSTM))
    return new_prediction_LSTM
