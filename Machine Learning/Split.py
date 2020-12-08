# -*- coding: utf-8 -*-

import pandas as pd

def splitIp(df, column_to_split):
    df[[column_to_split+'_1',column_to_split+'_2',column_to_split+'_3', column_to_split+'_4']] = df[column_to_split].str.split(".",expand=True,)

    del df[column_to_split]
    '''
    df.insert(column_index, df[column_to_split+'_1'], int)
    df.insert(column_index+1, df[column_to_split+'_2'], int)
    df.insert(column_index+2, df[column_to_split+'_3'], int)
    df.insert(column_index+3, df[column_to_split+'_4'], int)
'''
    return df
