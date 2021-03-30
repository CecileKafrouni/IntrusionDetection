# -*- coding: utf-8 -*-

'''
    ----------------- Retirer aleatoirement des lignes du csv -----------------
'''


import pandas as pd
import numpy as np


def removeRows(df):
    
    list_columns = ['SourceIP','DestinationIP', 'SourcePort','DestinationPort', 'Duration', 'FlowBytesSent', 'FlowSentRate', 'FlowBytesReceived', 'FlowReceivedRate',
'PacketLengthVariance', 'PacketLengthStandardDeviation', 'PacketLengthMean', 'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian',
'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation', 'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean', 'PacketTimeMedian',
'PacketTimeMode', 'PacketTimeSkewFromMedian', 'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation', 'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation',
'ResponseTimeTimeMean', 'ResponseTimeTimeMedian', 'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian', 'ResponseTimeTimeSkewFromMode', 'ResponseTimeTimeCoefficientofVariation']
    
    random_numbers = []
    random_numbers = np.random.randint(len(df), size=(5))
    
    df_copy = pd.DataFrame(columns=list_columns)
    
    j = 0
    
    for i in random_numbers:
        df_copy.loc[j]= df.loc[i]
        df.drop(index = i, inplace = True)
        j += 1
    
    
    return df_copy,df
    

