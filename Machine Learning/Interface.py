# -*- coding: utf-8 -*-

'''
----------------------- INTERFACE ---------------------------
'''

import PySimpleGUI as sg
import pandas as pd

def interface(df, target):
    df = df.drop([target], axis = 1)
    liste_colonne = list(df.columns)
    print(len(liste_colonne))
    
    sg.theme('DarkBlue3')   
    
    left_col = []
    right_col = []
    #left_col.append([sg.Text('Veuillez rentrer vos informations :')])
        
    #for i in range(0,len(liste_colonne)-1):
    for i in range(0,int(len(liste_colonne)/2)+1):
        left_col.append([sg.Text(liste_colonne[i], size=(30, 1)), 
                       sg.InputText(default_text = 0,size=(10,1))])
     
        
    for i in range(int(len(liste_colonne)/2)+1,len(liste_colonne)):
        right_col.append([sg.Text(liste_colonne[i], size=(30, 1)), 
                       sg.InputText(default_text = 0,size=(10,1))])
        
    
    frame_layout = [
                    [sg.Column(left_col, element_justification='c'), 
                     sg.VerticalSeparator(pad=None),
                     sg.Column(right_col, element_justification='c')]
                    ]

    #frame_layout.append([sg.Button('Ok'), sg.Button('Cancel')])

    layout = [
              [sg.Frame('Veuillez rentrer vos informations', frame_layout, font=15)],
              [sg.Button('Ok'), sg.Button('Cancel')]
             ]
   
    
    window = sg.Window('Informations pour prediction', layout, size=(750,520))
    event, values = window.read()
    while (event != 'Ok'):
        
        if event == sg.WIN_CLOSED or event == 'Cancel' : 
            break
       
        X_new = []
        for i in range (0, len(left_col)):
            X_new.append(values[i])
        for i in range (0, len(right_col)):
            X_new.append(values[i])
        print(X_new)
    
    features=[]
    for i in range(0,len(liste_colonne)):
        features.append(liste_colonne[i])
        
    print(features)

    df_new = pd.DataFrame([X_new], columns=features)
    df_new.to_csv('df_new.csv', sep=';', index = False)
    
    
    window.close()
    
    return df_new
    
if __name__ == "__main__":
    interface()
    
def result(pred_DTC_DoH,pred_RFC_DoH, pred_XGB_DoH):
    
    if(pred_DTC_DoH == 1.0):
        resultat_DTC_DoH = 'nonDoH'
    else:
        resultat_DTC_DoH = 'DoH'
    
    if(pred_RFC_DoH == 1.0):
        resultat_RFC_DoH = 'nonDoH'
    else:
        resultat_RFC_DoH = 'DoH'
    
    if(pred_XGB_DoH == 1.0):
        resultat_XGB_DoH = 'nonDoH'
    else:
        resultat_XGB_DoH = 'DoH'
       
    sg.popup('Resultat', 'Le resultat pour le DTC :{}'.format(resultat_DTC_DoH),
             'Le resultat pour le RFC :{}'.format(resultat_RFC_DoH), 
             'Le resultat pour le XGB :{}'.format(resultat_XGB_DoH))
    
    