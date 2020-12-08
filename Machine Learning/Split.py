# -*- coding: utf-8 -*-

import pandas as pd

def splitIp(df, column_to_split):
    df[[column_to_split+'_1',column_to_split+'_2',column_to_split+'_3', column_to_split+'_4']] = df[column_to_split].str.split(".",expand=True,)
    df[column_to_split+'_1'] = pd.to_numeric(df[column_to_split+'_1'])
    df[column_to_split+'_2'] = pd.to_numeric(df[column_to_split+'_2'])
    df[column_to_split+'_3'] = pd.to_numeric(df[column_to_split+'_3'])
    df[column_to_split+'_4'] = pd.to_numeric(df[column_to_split+'_4'])

    del df[column_to_split]
    return df


