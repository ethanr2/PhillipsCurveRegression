# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:25:43 2019

@author: ethan
"""

from datetime import date
from datetime import datetime as dt
from time import time

import numpy as np
import pandas as pd
from scipy import stats

from bokeh.io import export_png,output_file,show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter, LabelSet,ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.layouts import row, column


def load_NGDP_df():
    def col_filt(col):
        return not (col == 'WEO_Country_Code' or col == 'ISOAlpha_3Code')
    dfs = pd.read_excel('data/WEOhistorical.xlsx', sheet_name = [1,2], 
                         usecols = col_filt, index_col = [0,1], 
                         na_values = '.')

    dfs[2].columns = dfs[1].columns
    df = (dfs[1].astype(float) + dfs[2].astype(float))/100 + 1
    df = df.reset_index()
    return df.set_index(['country', 'year'], drop=False)
df_perm = load_NGDP_df()
print(df_perm)

#just look at us and israel for now
countries = ['United States', 'Israel']

ngdp = pd.read_excel('data\pwt91.xlsx',  sheet_name = 2)
ngdp = ngdp.loc[ngdp['year'] >= 1990, ['country','currency_unit','year','cgdpo','xr','pl_gdpo']]
bools = ngdp['country'].apply(lambda x: x in countries)
ngdp_perm = ngdp.loc[bools, :]
print(ngdp)
# <codecell>

df = df_perm.loc[countries, :]
ngdp = ngdp_perm.copy()
def impute(df):
    def convert(col, country):
        year = int(col.name[1:5])
        yeardiff = df.loc[country,'year']  - year
        col = col[yeardiff >=0]
        new_col = list(col[yeardiff < 6])
        new_col.insert(0, country)
        ind = ['country', 0,1,2,3,4,5]
        return pd.Series(new_col, index = ind)
    data = []
    for country in countries:
        temp_df = df.loc[country,:]
        temp = temp_df.iloc[:,2:].apply(lambda col: convert(col, country))
        temp.loc['obs',:] = temp.columns.values
        temp.columns = range(59)
        if isinstance(data, list):
            data = temp.T
        else:
            data = data.append(temp.T, ignore_index = True)
        print(data)
    return data
data = impute(df)
data['year'] = data['obs'].apply(lambda x: int(x[1:5]))
data = data.loc[data['year'] < 2018,:]
g = data.groupby(['country','year'])
def take_mean(year_df):
    temp = year_df.loc[:, list(range(5))]
    a_row = temp.iloc[0,:]
    b_row = temp.iloc[1,:]
    return a_row*b_row
   
levels = np.sqrt(g.apply(take_mean))
print(levels)

ngdp['ngdp'] = ngdp['cgdpo'] * ngdp['xr'] * ngdp['pl_gdpo']
ngdp= ngdp.set_index(['country', 'year'])

    

test = levels.apply(lambda x: ngdp.loc[x.name, 'ngdp']*x, axis = 1)
print(test)








