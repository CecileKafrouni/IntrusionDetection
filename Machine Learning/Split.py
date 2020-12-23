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


def IP2Int(df,column):
    i=0
    for values in df[column].values:
        o = list(map(int, values.split('.')))
        res = (16777216 * o[0]) + (65536 * o[1]) + (256 * o[2]) + o[3]
        df[column][i] = res
        i+=1
    return df


def Int2IP(ipnum):
    o1 = int(ipnum / 16777216) % 256
    o2 = int(ipnum / 65536) % 256
    o3 = int(ipnum / 256) % 256
    o4 = int(ipnum) % 256
    return '%(o1)s.%(o2)s.%(o3)s.%(o4)s' % locals()


