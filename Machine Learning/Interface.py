# -*- coding: utf-8 -*-

'''
----------------------- INTERFACE ---------------------------
'''

import PySimpleGUI as sg
import pandas as pd


def beginInterface():
    sg.theme('DarkBlue3')
    
    
    frame_layout =  [[sg.Text("Pour utiliser l'application avec des données de test, choisissez ' Test '. \nSi vous voulez insérer vous-même les valeurs, cliquez sur ' New '.")],
                       [sg.Radio('Test', "RADIO1", default=True),
    sg.Radio('New', "RADIO1")]]
    
    
    layout = [
              [sg.Frame('Veuillez sélectionner une case', frame_layout, font=15)],
              [sg.Button('Ok'), sg.Button('Cancel')]
             ]
   
    window = sg.Window('TEST', layout, size=(500,150))
    
    radio_value=''

    
    while (True):
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': 
            radio_value='New'
            window.close()
            break
        
        elif event == 'Ok':
            print(values)
            if values[0] == True:
                radio_value = 'Test'
            
            elif values[1] == True:
                radio_value = 'New'
            
            window.close()
            break
        
    return radio_value

def interface(df):
    #df = df.drop([target], axis = 1)
    #liste_colonne = list(df.columns)
    liste_colonne = ['SourceIP','DestinationIP', 'SourcePort','DestinationPort', 'Duration', 'FlowBytesSent', 'FlowSentRate', 'FlowBytesReceived', 'FlowReceivedRate',
'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian',
'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian',
'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation',
'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']
    
    
    
    sg.theme('DarkBlue3')   
    
    frame_layout = []
    
    if not df.empty:
        for i in range(0,len(liste_colonne)):
            frame_layout.append([sg.Text(liste_colonne[i], size=(30, 1)), 
                           sg.InputText(default_text = df[liste_colonne[i]].values[0],size=(10,1))])
    else:
        for i in range(0,len(liste_colonne)):
            frame_layout.append([sg.Text(liste_colonne[i], size=(30, 1)), 
                           sg.InputText(default_text = 0,size=(10,1))])
    frame_layout = [
                    [sg.Column(frame_layout[0:int(len(liste_colonne)/2)+1], element_justification='c'), 
                     sg.VerticalSeparator(pad=None),
                     sg.Column(frame_layout[int(len(liste_colonne)/2)+1:len(liste_colonne)], element_justification='c')]
                    ]

    layout = [
              [sg.Frame('Veuillez rentrer vos informations', frame_layout, font=15)],
              [sg.Button('Ok'), sg.Button('Cancel')]
             ]
   
    
    window = sg.Window('Informations pour prediction', layout, size=(750,520))
    
    button_value=''

    
    while (True):
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel': 
            button_value='Cancel'
            window.close()
            break
        
        elif event == 'Ok':
            X_new = []
            for j in range (0, len(liste_colonne)):
                X_new.append(values[j])
            features=[]
            for i in range(0,len(liste_colonne)):
                features.append(liste_colonne[i])
                
            df_new = pd.DataFrame([X_new], columns=features)
            df_new.to_csv('df_new.csv', sep=';', index = False)
            button_value='Ok'
            window.close()
            break
        
    return button_value
    

if __name__ == "__main__":
    interface()
    
def result(pred_DTC_DoH,pred_RFC_DoH, pred_XGB_DoH, pred_GNB_DoH,pred_KNN_DoH,pred_Conv1D_Model_Intrusion,pred_Conv2D_Model_Intrusion):
    
    compteur=0
    
    if(pred_DTC_DoH == 1.0):
        resultat_DTC_DoH = 'DoH'
        compteur+=1
    else:
        resultat_DTC_DoH = 'nonDoH'
    
    if(pred_RFC_DoH == 1.0):
        resultat_RFC_DoH = 'DoH'
        compteur+=1
    else:
        resultat_RFC_DoH = 'nonDoH'
            
    if(pred_XGB_DoH == 1.0):
        resultat_XGB_DoH = 'DoH'
        compteur+=1
    else:
        resultat_XGB_DoH = 'nonDoH'
       
        
    if(pred_GNB_DoH == 1.0):
        resultat_GNB_DoH = 'DoH'
        compteur+=1
    else:
        resultat_GNB_DoH = 'nonDoH'
        
    if(pred_KNN_DoH == 1.0):
        resultat_KNN_DoH = 'DoH'
        compteur+=1
    else:
        resultat_KNN_DoH = 'nonDoH'
    
    
    if(compteur >= 2):
        sg.popup('Resultat DoH', 'Le resultat pour le DTC :{}'.format(resultat_DTC_DoH),
                 'Le resultat pour le RFC :{}'.format(resultat_RFC_DoH), 
                 'Le resultat pour le XGB :{}'.format(resultat_XGB_DoH),
                 
                 'Le resultat pour le GNB :{}'.format(resultat_GNB_DoH),
                 'Le resultat pour le KNN :{}'.format(resultat_KNN_DoH))
        result_intrusion(pred_Conv1D_Model_Intrusion,pred_Conv2D_Model_Intrusion)
    else:
        sg.popup('Resultat DoH', 'Le resultat pour le DTC :{}'.format(resultat_DTC_DoH),
                 'Le resultat pour le RFC :{}'.format(resultat_RFC_DoH), 
                 'Le resultat pour le XGB :{}'.format(resultat_XGB_DoH),
                 
                 'Le resultat pour le GNB :{}'.format(resultat_GNB_DoH),
                 'Le resultat pour le KNN :{}'.format(resultat_KNN_DoH))
    
def result_intrusion(pred_Conv1D_Model_Intrusion,pred_Conv2D_Model_Intrusion):
    
        
    if(pred_Conv1D_Model_Intrusion == 1.0):
        resultat_Conv1D_Model_Intrusion = 'Intrusion'
    else:
        resultat_Conv1D_Model_Intrusion = 'Begnin'
        
    if(pred_Conv2D_Model_Intrusion == 1.0):
        resultat_Conv2D_Model_Intrusion = 'Intrusion'
    else:
        resultat_Conv2D_Model_Intrusion = 'Begnin'
    
        
    sg.popup('Resultat Intrusion',
             'Le resultat pour le modele Conv1D : {}'.format(resultat_Conv1D_Model_Intrusion),
             'Le resultat pour le modele Conv2D : {}'.format(resultat_Conv2D_Model_Intrusion))