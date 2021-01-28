import seaborn as sns
import pandas as pd

def remove_outliers(df, colonne):
    
    sns.boxplot(x=df[colonne], width=0.5)
    
    Q1 = df[colonne].quantile(0.25)
    Q3 = df[colonne].quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    
    print('Shape of the df with outliers : {}'.format(df.shape))
    
    df[colonne] = df[colonne][~((df[colonne] < (Q1 - 1.5 * IQR)) |(df[colonne] > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    print('Shape of the df without outliers : {}'.format(df.shape))
    
    return df[colonne]

def remove_outliers_features(df) :
    for colonne in df.columns:
        if(colonne != 'Intrusion' 
           and colonne != 'DestinationIP'
           and colonne != 'DestinationIP'
           and colonne != 'SourcePort'
           and colonne != 'DestinationPort'):
           df[colonne] = remove_outliers(df, colonne)
           
    return df

df = pd.read_csv("Intrusion/total_csv_copy_Intrusion.csv", sep=';')

df_outliers_removed = remove_outliers_features(df)

