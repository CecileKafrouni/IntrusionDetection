import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort

import pandas as pd
import Modeles_ML as ml
import Modeles_ML_autres as ml_bis
import Modeles_DL as dl
import pickle
from keras.models import load_model
import Preparation_CSV as pc

import Normalisation_donnees as nor

liste_colonne = ['SourceIP','DestinationIP', 'SourcePort','DestinationPort', 'Duration', 'FlowBytesSent', 'FlowSentRate', 'FlowBytesReceived', 'FlowReceivedRate',
'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian',
'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian',
'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation',
'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']


#Loading ML models for DoH

filename_DTC_DoH = 'DoH/finalized_model_DTC_DoH.sav'
filename_RFC_DoH = 'DoH/finalized_model_RFC_DoH.sav'
filename_XGB_DoH = 'DoH/finalized_model_XGB_DoH.sav'

filename_GNB_DoH = 'DoH/finalized_model_GNB_DoH.sav'
filename_KNN_DoH = 'DoH/finalized_model_KNN_DoH.sav'
filename_SVM_DoH = 'DoH/finalized_model_SVM_DoH.sav'

loaded_model_DTC_DoH = pickle.load(open(filename_DTC_DoH, 'rb'))
loaded_model_RFC_DoH = pickle.load(open(filename_RFC_DoH, 'rb'))
loaded_model_XGB_DoH = pickle.load(open(filename_XGB_DoH, 'rb'))

#loaded_model_GNB_DoH = pickle.load(open(filename_GNB_DoH, 'rb'))
#loaded_model_KNN_DoH = pickle.load(open(filename_KNN_DoH, 'rb'))

#Loading DL models for Intrusion

#filename_Simple_DL_Model_Intrusion = 'Intrusion/finalized_model_Simple_DL_Model_Intrusion.h5'
#loaded_model_Simple_DL_Model_Intrusion = load_model(filename_Simple_DL_Model_Intrusion)

filename_Conv1D_Model_Intrusion = 'Intrusion/Conv1D.h5'
filename_Conv2D_Model_Intrusion = 'Intrusion/Conv2D.h5'
#filename_LSTM_Model_Intrusion = 'Intrusion/LSTM.h5'

loaded_model_Conv1D_Model_Intrusion = load_model(filename_Conv1D_Model_Intrusion)
loaded_model_Conv2D_Model_Intrusion = load_model(filename_Conv2D_Model_Intrusion)
#loaded_model_LSTM_Model_Intrusion = load_model(filename_LSTM_Model_Intrusion)


original_dataset=pd.read_csv('DoH/total_csv_copy_DoH.csv', sep=';')

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_post(post_id):
    conn = get_db_connection()
    post = conn.execute('SELECT * FROM posts WHERE id = ?',
                        (post_id,)).fetchone()
    conn.close()
    if post is None:
        abort(404)
    return post

app = Flask(__name__)
app.config['SECRET_KEY'] = 'azerty1234'

@app.route('/')
def index():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    return render_template('index.html', posts=posts)



@app.route('/<int:post_id>')
def post(post_id):
    post = get_post(post_id)
    return render_template('post.html', post=post)


@app.route('/test', methods=('GET', 'POST'))
def test():
    if request.method == 'POST':
        title = request.form['title']
        
        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO posts (title,ok) VALUES (?,?)',
                         (title, 'ok'))
            
            id = conn.execute('SELECT id FROM posts WHERE title=? AND ok=?', (title, 'ok')).fetchone()[0]
            
            print(id)
            print('type de l id ',type(id))
            conn.commit()
            conn.close()
            
            predict_test(id)
            
            return redirect(url_for('result_test', id=str(id)))
            
    return render_template('test.html')

@app.route('/new', methods=('GET', 'POST'))
def new():
    if request.method == 'POST':
        title = request.form['title']
        
        
        SourceIP = request.form['SourceIP']
        DestinationIP = request.form['DestinationIP']
        SourcePort = request.form['SourcePort']
        DestinationPort = request.form['DestinationPort']
        Duration = request.form['Duration']
        FlowBytesSent = request.form['FlowBytesSent']
        FlowSentRate = request.form['FlowSentRate']
        FlowBytesReceived = request.form['FlowBytesReceived']
        FlowReceivedRate = request.form['FlowReceivedRate']
        PacketLengthVariance = request.form['PacketLengthVariance']
        PacketLengthStandardDeviation = request.form['PacketLengthStandardDeviation']
        PacketLengthMean = request.form['PacketLengthMean']
        PacketLengthMedian = request.form['PacketLengthMedian']
        PacketLengthMode = request.form['PacketLengthMode']
        PacketLengthSkewFromMedian = request.form['PacketLengthSkewFromMedian']
        PacketLengthSkewFromMode = request.form['PacketLengthSkewFromMode']
        PacketLengthCoefficientofVariation = request.form['PacketLengthCoefficientofVariation']
        PacketTimeVariance = request.form['PacketTimeVariance']
        PacketTimeStandardDeviation = request.form['PacketTimeStandardDeviation']
        PacketTimeMean = request.form['PacketTimeMean']
        PacketTimeMedian = request.form['PacketTimeMedian']
        PacketTimeMode = request.form['PacketTimeMode']
        PacketTimeSkewFromMedian = request.form['PacketTimeSkewFromMedian']
        PacketTimeSkewFromMode = request.form['PacketTimeSkewFromMode']
        PacketTimeCoefficientofVariation = request.form['PacketTimeCoefficientofVariation']
        ResponseTimeTimeVariance = request.form['ResponseTimeTimeVariance']
        ResponseTimeTimeStandardDeviation = request.form['ResponseTimeTimeStandardDeviation']
        ResponseTimeTimeMean = request.form['ResponseTimeTimeMean']
        ResponseTimeTimeMedian = request.form['ResponseTimeTimeMedian']
        ResponseTimeTimeMode = request.form['ResponseTimeTimeMode']
        ResponseTimeTimeSkewFromMedian = request.form['ResponseTimeTimeSkewFromMedian']
        ResponseTimeTimeSkewFromMode = request.form['ResponseTimeTimeSkewFromMode']
        ResponseTimeTimeCoefficientofVariation = request.form['ResponseTimeTimeCoefficientofVariation']

        
        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('INSERT INTO posts (title,ok, SourceIP, DestinationIP, SourcePort,DestinationPort, Duration , FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,PacketLengthSkewFromMode, PacketLengthCoefficientofVariation, PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation, ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean, ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian, ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation) VALUES ( ?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?)',
                         (title, 'ok', SourceIP, DestinationIP, SourcePort,DestinationPort, Duration , FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,PacketLengthSkewFromMode, PacketLengthCoefficientofVariation, PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation, ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean, ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian, ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation))
            
            
            id = conn.execute('SELECT id FROM posts WHERE title=? AND SourceIP=? AND DestinationIP=? AND SourcePort=? AND DestinationPort=? AND Duration=? AND FlowBytesSent=? AND FlowSentRate=? AND FlowBytesReceived=? AND FlowReceivedRate=? AND PacketLengthVariance=? AND PacketLengthStandardDeviation=? AND PacketLengthMean=? AND PacketLengthMedian=? AND PacketLengthMode=? AND PacketLengthSkewFromMedian=? AND PacketLengthSkewFromMode=? AND PacketLengthCoefficientofVariation=? AND PacketTimeVariance=? AND PacketTimeStandardDeviation=? AND PacketTimeMean=? AND PacketTimeMedian=? AND PacketTimeMode=? AND PacketTimeSkewFromMedian=? AND PacketTimeSkewFromMode=? AND PacketTimeCoefficientofVariation=? AND ResponseTimeTimeVariance=? AND ResponseTimeTimeStandardDeviation=? AND ResponseTimeTimeMean=? AND ResponseTimeTimeMedian=? AND ResponseTimeTimeMode=? AND ResponseTimeTimeSkewFromMedian=? AND ResponseTimeTimeSkewFromMode=? AND ResponseTimeTimeCoefficientofVariation=?',
                         (title, SourceIP, DestinationIP, SourcePort,DestinationPort, Duration , FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,PacketLengthSkewFromMode, PacketLengthCoefficientofVariation, PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation, ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean, ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian, ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation)).fetchone()[0]
        
           
            conn.commit()
            #conn.close()
            
            df_test = pd.DataFrame([[SourceIP, DestinationIP, SourcePort,DestinationPort, Duration , FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,PacketLengthSkewFromMode, PacketLengthCoefficientofVariation, PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation, ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean, ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian, ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation]],columns = liste_colonne)
            df_test.to_csv('df/df_test_'+str(id)+'.csv', sep=';')
            
            
            for colonne in df_test.columns:
                if colonne == '' or colonne == 'Timestamp' or colonne == 'TimeStamp' or colonne == 'index' or colonne == 'Label' or colonne == 'Unnamed: 0':
                    del df_test[colonne]
            for colonne in df_test.columns:
                if type(df_test[colonne][0]) == str and colonne != 'SourceIP' and colonne != 'DestinationIP':
                    df_test[colonne][0] = 0
            
            conn = get_db_connection()
            conn.execute('UPDATE posts SET SourceIP =?, DestinationIP=? WHERE id=?',
                                 (df_test['SourceIP'][0], df_test['DestinationIP'][0],id))
            
            
            if type(df_test['SourceIP']) != float:
                df_test = pc.IP2Int(df_test, 'SourceIP')
            if type(df_test['DestinationIP']) != float:
                df_test = pc.IP2Int(df_test, 'DestinationIP')
            df_test = pc.nettoyage(df_test)
            df_test=nor.NormalizeNewValues(original_dataset, df_test)
            
            
            for colonne in df_test.columns:
                if(colonne != 'SourceIP' and colonne != 'DestinationIP'):
                    df_test[colonne] = float(df_test[colonne])
                    conn.execute('UPDATE posts SET '+colonne+' =? WHERE id=?',
                                 (df_test[colonne][0], id))
                
               
            conn.commit()
             # Stocke predictions
            pred_DTC_DoH = ml.DTC_Prediction(df_test, loaded_model_DTC_DoH)
            pred_XGB_DoH = ml.XGB_Prediction(df_test, loaded_model_XGB_DoH)
            pred_RFC_DoH = ml.RFC_Prediction(df_test, loaded_model_RFC_DoH)
            
            #df_test_norm = nor.NormalizeDataset(df_test)
            #pred_GNB_DoH = ml_bis.GNB_Prediction(df_test_norm, loaded_model_GNB_DoH)
            pred_GNB_DoH=0
            #pred_KNN_DoH = ml_bis.KNN_Prediction(df_test_norm, loaded_model_KNN_DoH)
            pred_KNN_DoH = 0
            
            pred_Conv1D_Model_Intrusion = dl.Conv1D_Prediction(df_test, loaded_model_Conv1D_Model_Intrusion)
            pred_Conv2D_Model_Intrusion = dl.Conv2D_Prediction(df_test, loaded_model_Conv2D_Model_Intrusion)
           
            #conn = get_db_connection()
            conn.execute('UPDATE posts SET pred_DTC_DoH = ?, pred_RFC_DoH = ?, pred_XGB_DoH = ?, pred_KNN_DoH = ?,  pred_GNB_DoH = ?,  pred_CNN_1D_Intrusion = ?, pred_CNN_2D_Intrusion = ?'
                         ' WHERE id = ?',
                         (int(pred_DTC_DoH),int(pred_RFC_DoH),int(pred_XGB_DoH),int(pred_KNN_DoH),int(pred_GNB_DoH),int(pred_Conv1D_Model_Intrusion),int(pred_Conv2D_Model_Intrusion), id))
            
            
            
            compteur_DoH=0
            compteur_Intrusion=0
            
            if(pred_DTC_DoH == 1.0):
                compteur_DoH+=1
           
            if(pred_RFC_DoH == 1.0):
                compteur_DoH+=1
           
            if(pred_XGB_DoH == 1.0):
                compteur_DoH+=1
            
            if(pred_GNB_DoH == 1.0):
                compteur_DoH+=1
           
            if(pred_KNN_DoH == 1.0):
                compteur_DoH+=1
            
            if(pred_Conv1D_Model_Intrusion == 1.0):
                compteur_Intrusion+=1
                
            if(pred_Conv2D_Model_Intrusion == 1.0):
                compteur_Intrusion+=1
            
            print('compteur_DoH',compteur_DoH)
            print('compteur_Intrusion',compteur_Intrusion)
            
            if(compteur_DoH >= 2):
                conn.execute('UPDATE posts SET Label_DoH  = ?, Nb_model_DoH = ?'
                         ' WHERE id = ?',
                         ('DoH',str(compteur_DoH), id))
                
                if(compteur_Intrusion==2):
                    conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion = ?'
                         ' WHERE id = ?',
                         ('Intrusif',str(compteur_Intrusion), id))
                else:
                     conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion = ?'
                         ' WHERE id = ?',
                         ('Benin',str(compteur_Intrusion), id))
               
            else:
                conn.execute('UPDATE posts SET Label_DoH  = ?, Nb_model_DoH = ?'
                         ' WHERE id = ?',
                         ('non DoH',str(compteur_DoH), id))
                if(compteur_Intrusion==2):
                    conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion= ?'
                         ' WHERE id = ?',
                         ('Intrusif',str(compteur_Intrusion), id))
                else:
                     conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion= ?'
                         ' WHERE id = ?',
                         ('Benin',str(compteur_Intrusion), id))
            
            conn.commit()
            conn.close()
            
            
            
            return redirect(url_for('result_new', id=id))

    return render_template('new.html')

@app.route('/<int:id>/result_new', methods=('GET', 'POST'))
def result_new(id):
    post = get_post(id)
    
    return render_template('result_new.html', post=post)

def predict_test(id):
    if request.method == 'POST':
        csv = request.form['file']
        print(csv)
        df_test=pd.read_csv('Tests_interface/'+csv, sep=';')
        
        #df_test = pd.read_csv('df_test.csv', sep=';')
        
        for colonne in df_test.columns:
            if colonne == '' or colonne == 'Timestamp' or colonne == 'TimeStamp' or colonne == 'index' or colonne == 'Label' or colonne == 'Unnamed: 0':
                del df_test[colonne]
        for colonne in df_test.columns:
            if type(df_test[colonne][0]) == str and colonne != 'SourceIP' and colonne != 'DestinationIP':
                df_test[colonne][0] = 0
        
        conn = get_db_connection()
        conn.execute('UPDATE posts SET SourceIP =?, DestinationIP=? WHERE id=?',
                             (df_test['SourceIP'][0], df_test['DestinationIP'][0],id))
        
        
        if type(df_test['SourceIP']) != float:
            df_test = pc.IP2Int(df_test, 'SourceIP')
        if type(df_test['DestinationIP']) != float:
            df_test = pc.IP2Int(df_test, 'DestinationIP')
        df_test = pc.nettoyage(df_test)
        df_test=nor.NormalizeNewValues(original_dataset, df_test)
        
        
        for colonne in df_test.columns:
            if(colonne != 'SourceIP' and colonne != 'DestinationIP'):
                df_test[colonne] = float(df_test[colonne])
                conn.execute('UPDATE posts SET '+colonne+' =? WHERE id=?',
                             (df_test[colonne][0], id))
            
           
        conn.commit()
        
        #df_test = pd.read_csv('df/df_test_'+str(3)+'.csv', sep=';')
        
         # Stocke predictions
        pred_DTC_DoH = ml.DTC_Prediction(df_test, loaded_model_DTC_DoH)
        pred_XGB_DoH = ml.XGB_Prediction(df_test, loaded_model_XGB_DoH)
        pred_RFC_DoH = ml.RFC_Prediction(df_test, loaded_model_RFC_DoH)
        
        #df_test_norm = nor.NormalizeDataset(df_test)
        #pred_GNB_DoH = ml_bis.GNB_Prediction(df_test_norm, loaded_model_GNB_DoH)
        pred_GNB_DoH=0
        #pred_KNN_DoH = ml_bis.KNN_Prediction(df_test_norm, loaded_model_KNN_DoH)
        pred_KNN_DoH = 0
        
        pred_Conv1D_Model_Intrusion = dl.Conv1D_Prediction(df_test, loaded_model_Conv1D_Model_Intrusion)
        pred_Conv2D_Model_Intrusion = dl.Conv2D_Prediction(df_test, loaded_model_Conv2D_Model_Intrusion)
       
        conn = get_db_connection()
        conn.execute('UPDATE posts SET pred_DTC_DoH = ?, pred_RFC_DoH = ?, pred_XGB_DoH = ?, pred_KNN_DoH = ?,  pred_GNB_DoH = ?,  pred_CNN_1D_Intrusion = ?, pred_CNN_2D_Intrusion = ?'
                     ' WHERE id = ?',
                     (int(pred_DTC_DoH),int(pred_RFC_DoH),int(pred_XGB_DoH),int(pred_KNN_DoH),int(pred_GNB_DoH),int(pred_Conv1D_Model_Intrusion),int(pred_Conv2D_Model_Intrusion), id))
        
        
        compteur_DoH=0
        compteur_Intrusion=0
        
        if(pred_DTC_DoH == 1.0):
            compteur_DoH+=1
       
        if(pred_RFC_DoH == 1.0):
            compteur_DoH+=1
       
        if(pred_XGB_DoH == 1.0):
            compteur_DoH+=1
        
        if(pred_GNB_DoH == 1.0):
            compteur_DoH+=1
       
        if(pred_KNN_DoH == 1.0):
            compteur_DoH+=1
        
        if(pred_Conv1D_Model_Intrusion == 1.0):
            compteur_Intrusion+=1
            
        if(pred_Conv2D_Model_Intrusion == 1.0):
            compteur_Intrusion+=1
        
        print('compteur_DoH',compteur_DoH)
        print('compteur_Intrusion',compteur_Intrusion)
        
        if(compteur_DoH >= 2):
            conn.execute('UPDATE posts SET Label_DoH  = ?, Nb_model_DoH = ?'
                     ' WHERE id = ?',
                     ('DoH',str(compteur_DoH), id))
            
            if(compteur_Intrusion==2):
                conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion = ?'
                     ' WHERE id = ?',
                     ('Intrusif',str(compteur_Intrusion), id))
            else:
                 conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion = ?'
                     ' WHERE id = ?',
                     ('Benin',str(compteur_Intrusion), id))
           
        else:
            conn.execute('UPDATE posts SET Label_DoH  = ?, Nb_model_DoH = ?'
                     ' WHERE id = ?',
                     ('non DoH',str(compteur_DoH), id))
            if(compteur_Intrusion==2):
                conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion= ?'
                     ' WHERE id = ?',
                     ('Intrusif',str(compteur_Intrusion), id))
            else:
                 conn.execute('UPDATE posts SET Label_Intrusion  = ?, Nb_model_Intrusion= ?'
                     ' WHERE id = ?',
                     ('Benin',str(compteur_Intrusion), id))
        
        conn.commit()
        conn.close()


@app.route('/<int:id>/result_test', methods=('GET', 'POST'))
def result_test(id):
    post = get_post(id)
    
    return render_template('result_test.html', post=post)

@app.route('/<int:id>/edit', methods=('GET', 'POST'))
def edit(id):
    post = get_post(id)

    if request.method == 'POST':
        title = request.form['title']
      

        SourceIP = request.form['SourceIP']
        DestinationIP = request.form['DestinationIP']
        SourcePort = request.form['SourcePort']
        DestinationPort = request.form['DestinationPort']
        Duration = request.form['Duration']
        FlowBytesSent = request.form['FlowBytesSent']
        FlowSentRate = request.form['FlowSentRate']
        FlowBytesReceived = request.form['FlowBytesReceived']
        FlowReceivedRate = request.form['FlowReceivedRate']
        PacketLengthVariance = request.form['PacketLengthVariance']
        PacketLengthStandardDeviation = request.form['PacketLengthStandardDeviation']
        PacketLengthMean = request.form['PacketLengthMean']
        PacketLengthMedian = request.form['PacketLengthMedian']
        PacketLengthMode = request.form['PacketLengthMode']
        PacketLengthSkewFromMedian = request.form['PacketLengthSkewFromMedian']
        PacketLengthSkewFromMode = request.form['PacketLengthSkewFromMode']
        PacketLengthCoefficientofVariation = request.form['PacketLengthCoefficientofVariation']
        PacketTimeVariance = request.form['PacketTimeVariance']
        PacketTimeStandardDeviation = request.form['PacketTimeStandardDeviation']
        PacketTimeMean = request.form['PacketTimeMean']
        PacketTimeMedian = request.form['PacketTimeMedian']
        PacketTimeMode = request.form['PacketTimeMode']
        PacketTimeSkewFromMedian = request.form['PacketTimeSkewFromMedian']
        PacketTimeSkewFromMode = request.form['PacketTimeSkewFromMode']
        PacketTimeCoefficientofVariation = request.form['PacketTimeCoefficientofVariation']
        ResponseTimeTimeVariance = request.form['ResponseTimeTimeVariance']
        ResponseTimeTimeStandardDeviation = request.form['ResponseTimeTimeStandardDeviation']
        ResponseTimeTimeMean = request.form['ResponseTimeTimeMean']
        ResponseTimeTimeMedian = request.form['ResponseTimeTimeMedian']
        ResponseTimeTimeMode = request.form['ResponseTimeTimeMode']
        ResponseTimeTimeSkewFromMedian = request.form['ResponseTimeTimeSkewFromMedian']
        ResponseTimeTimeSkewFromMode = request.form['ResponseTimeTimeSkewFromMode']
        ResponseTimeTimeCoefficientofVariation = request.form['ResponseTimeTimeCoefficientofVariation']



        if not title:
            flash('Title is required!')
        else:
            conn = get_db_connection()
            conn.execute('UPDATE posts SET title = ?, SourceIP = ?, DestinationIP = ?, SourcePort = ?,DestinationPort = ?, Duration = ? , FlowBytesSent = ?, FlowSentRate = ?, FlowBytesReceived = ?, FlowReceivedRate = ?,PacketLengthVariance = ?, PacketLengthStandardDeviation = ?, PacketLengthMean = ?, PacketLengthMedian = ?, PacketLengthMode = ?, PacketLengthSkewFromMedian = ?,PacketLengthSkewFromMode = ?, PacketLengthCoefficientofVariation = ?, PacketTimeVariance = ?, PacketTimeStandardDeviation = ?, PacketTimeMean = ?, PacketTimeMedian = ?, PacketTimeMode = ?,PacketTimeSkewFromMedian = ?, PacketTimeSkewFromMode = ?, PacketTimeCoefficientofVariation = ?, ResponseTimeTimeVariance = ?, ResponseTimeTimeStandardDeviation = ?,ResponseTimeTimeMean = ?, ResponseTimeTimeMedian = ?, ResponseTimeTimeMode = ?, ResponseTimeTimeSkewFromMedian = ?, ResponseTimeTimeSkewFromMode = ?, ResponseTimeTimeCoefficientofVariation = ?'
                         ' WHERE id = ?',
                         (title, SourceIP, DestinationIP, SourcePort,DestinationPort, Duration , FlowBytesSent, FlowSentRate, FlowBytesReceived, FlowReceivedRate,PacketLengthVariance, PacketLengthStandardDeviation, PacketLengthMean, PacketLengthMedian, PacketLengthMode, PacketLengthSkewFromMedian,PacketLengthSkewFromMode, PacketLengthCoefficientofVariation, PacketTimeVariance, PacketTimeStandardDeviation, PacketTimeMean, PacketTimeMedian, PacketTimeMode,PacketTimeSkewFromMedian, PacketTimeSkewFromMode, PacketTimeCoefficientofVariation, ResponseTimeTimeVariance, ResponseTimeTimeStandardDeviation,ResponseTimeTimeMean, ResponseTimeTimeMedian, ResponseTimeTimeMode, ResponseTimeTimeSkewFromMedian, ResponseTimeTimeSkewFromMode, ResponseTimeTimeCoefficientofVariation, id))
            conn.commit()
            conn.close()
            return redirect(url_for('index'))

    return render_template('edit.html', post=post)


@app.route('/<int:id>/delete', methods=('POST',))
def delete(id):
    post = get_post(id)
    conn = get_db_connection()
    conn.execute('DELETE FROM posts WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    flash('"{}" was successfully deleted!'.format(post['title']))
    return redirect(url_for('index'))
