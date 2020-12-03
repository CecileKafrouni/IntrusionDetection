# -*- coding: utf-8 -*-


'''
------------------------------ Preparation du CSV -----------------------------------
'''

# Librairies 
import pandas as pd
import numpy as np

# Les fonctions

def analyser_df(df):
    
    #Analyse rapide
    print("\nShape de la frame :\n", df.shape)
    print("\nNoms des colonnes :\n", df.columns)
    print("\nPremières lignes :\n", df.head())
    print("\nDernières lignes :\n", df.tail())
    
    #On regarde les differentes classes presentes dans la colonne label
    print("\n ELEMENTS DE LA COLONNE LABEL")
    print(df.Label.unique())
        
    #On compte le nombre de fois que chaque element revient
    print("\n ELEMENTS DE LA COLONNE LABEL AVEC NB DE FOIS QU'ILS REVIENNENT")
    print(df.Label.value_counts())

def rajouter_colonne(df, colonne) :
    
    df[colonne] = 0
    z = 0 

    for x in df[colonne]:
        if (colonne == 'Intrusion'):
            if df.Label[z] == 'Benign':
                df[colonne][z] = 0
            else :
                df[colonne][z] = 1       
            z = z+1
        else:
            if df.Label[z] == 'DoH':
                df[colonne][z] = 0
                
            else :
                df[colonne][z] = 1       
            z = z+1
 
def supprimer_colonne_vide(df) :
    for colonne in df:
        compteur = 0
        for valeur in df[colonne]:
            if valeur == 0:
                compteur +=1
                
        if compteur == len(df) or colonne == 'Timestamp' or colonne == 'TimeStamp' or colonne == 'index' or colonne == 'Label':
            del df[colonne]

def equilibrage_donnees(df, colonne):
    
    if(colonne == 'Intrusion'):
        df_intrusion = df[df['Label'] != 'Benign' ]
        df_intrusion = df_intrusion.reset_index()
        nb_Attack = len(df_intrusion)
        print('il y a ' + str(nb_Attack) +  ' attaques')
        
        df_benign = df[df['Label'] == 'Benign' ]
        df_benign = df_benign.reset_index()
        nb_benign = len(df_benign)
        print('il y a ' + str(nb_benign) +  ' cas benign')
        
        if nb_Attack < nb_benign :
                df_benign = df_benign[0:len(df_intrusion)]
                
        if nb_Attack > nb_benign :
                df_intrusion = df_intrusion[0:len(df_benign)]
                
        df_equilibre = pd.concat([df_intrusion,df_benign])
        df_equilibre = df_equilibre.reset_index(drop = True)
        
    else:
        df_nonDoH = df[df['Label'] != 'DoH' ]
        df_nonDoH = df_nonDoH.reset_index()
        nb_nonDoH = len(df_nonDoH)
        print('il y a ' + str(nb_nonDoH) +  ' non DoH')
        
        df_DoH = df[df['Label'] == 'DoH' ]
        df_DoH = df_DoH.reset_index()
        nb_DoH = len(df_DoH)
        print('il y a ' + str(nb_DoH) +  ' DoH')
        
        if nb_nonDoH < nb_DoH :
                df_DoH = df_DoH[0:len(df_nonDoH)]
                
        if nb_nonDoH > nb_DoH :
                df_nonDoH = df_nonDoH[0:len(df_DoH)]
    
        df_equilibre = pd.concat([df_nonDoH,df_DoH])
        df_equilibre = df_equilibre.reset_index(drop = True)
    
    print('Apres equilibrage on a :\n')
    print(df_equilibre['Label'].value_counts())
              
    return df_equilibre    



def nettoyage(df):
    result = df.copy()
    # On remplace els donnees Nan par O 
    result.fillna(0, inplace=True)
    
    
    # On arrondie les nb avec 5 chiffres apreès la virgule
    for colonne in result.columns:
        if(df[colonne].dtypes == 'float64'):
            a = np.array((result[colonne]))
            result[colonne] = np.around(a, decimals=5)
    
    return result


def Preparation_CSV(df, colonne):
    #analyser_df(df)
    df = equilibrage_donnees(df, colonne)
    rajouter_colonne(df, colonne)  
    supprimer_colonne_vide(df)
    
    df = nettoyage(df)
    
    df_copy = df.copy()
    
    return df_copy
