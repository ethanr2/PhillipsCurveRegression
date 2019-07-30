# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:20:16 2019

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

def make_sr():
    sr_df = pd.read_excel('data/NGDP.xlsx')
    
    cols = ['YEAR', 'QUARTER']
    for i in range(1,22):
        cols.append('NGDP' + str(i))
    
    return sr_df.loc[sr_df['YEAR'] >= 1992,cols]

def make_lr():
    rgdp = pd.read_excel('data/RGDP10_Level.xlsx').interpolate()
    cpi = pd.read_excel('data/CPI10_Level.xlsx')
    rgdp['CPI10'] = cpi['CPI10']
    lr_df = rgdp.loc[rgdp['YEAR'] >= 1992, :]
    
    def interp(row):
        ngdp =  (1+(row['RGDP10'] + row['CPI10'])/100)**(1/4)
        ngdp = pd.Series([ngdp]*20)
        return row[['YEAR','QUARTER']].append(ngdp.cumprod()[5:])
    lr_df = lr_df.apply(interp, axis = 1)
    
    cols = ['YEAR', 'QUARTER']
    for i in range(7,22):
        cols.append('NGDP' + str(i))
    cols = pd.Index(cols)
    lr_df.columns= cols
    return lr_df

def make_df():
    sr_df = make_sr()
    lr_df = make_lr()
    
    ind = sr_df[['YEAR','QUARTER']].apply(lambda x: dt(year = x['YEAR'], 
               month = (x['QUARTER']-1)*3+1, day = 1), axis = 1)
    ind = pd.DatetimeIndex(ind)
    sr_df = sr_df.iloc[:,2:].set_index(ind)
    lr_df = lr_df.iloc[:,2:].set_index(ind)
    
    sr_df.loc[:,lr_df.columns] = lr_df
    sr_df['NGDP1'] = sr_df['NGDP1'].shift(-1)
    def level(row):
        return row['NGDP1']*row.iloc[6:]
    sr_df.loc[:,lr_df.columns] = sr_df.apply(level, axis = 1)
    return sr_df

def make_u_ser():
    u_df = pd.read_csv('data/UNRATE.csv')
    ind = u_df['DATE'].apply(lambda x: dt.strptime(str(x),f'%Y-%m-%d' ))
    ind = pd.DatetimeIndex(ind)
    u_df = u_df.set_index(ind)
    return u_df['UNRATE'].astype(float)/100

def set_up(x, y, truncated = True, margins = None):
    if truncated: 
        b = (3 * y.min() - y.max())/2
    else:
        b = y.min()
    if margins == None:    
        xrng = (x.min(),x.max())
        yrng = (b,y.max())
    else:
        xrng = (x.min() - margins,x.max() + margins)
        yrng = (b - margins,y.max() + margins)
        
    x = x.dropna()
    y = y.dropna()
    
    return(x,y,xrng,yrng)
    
def chart_NGDP_ser(sticky_forecast):
    xdata, ydata, xrng, yrng = set_up(sticky_forecast.index,sticky_forecast['Forecast'])
    
    p = figure(width = 1000, height = 700,
               title="Sticky Forecast of NGDP" , 
               x_axis_label = 'Date', x_axis_type = 'datetime',
               y_axis_label = 'NGDP (Billions of US Dollars)', 
               y_range = yrng, x_range = xrng)

    p.line(xrng,[0,0], color = 'black')
    p.line(xdata,ydata, color = 'blue', legend = 'Sticky-Forecast of NGDP')
    p.line(xdata,sticky_forecast['NGDP1'], color = 'black', legend = 'Actual NGDP')
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = 'top_left'
    p.ygrid.grid_line_color = None
    p.yaxis.formatter=NumeralTickFormatter(format="$0,000")
    
    export_png(p,'images/NGDP_series.png')

    return p

def chart_gap_ser(df):
    xdata, ydata, xrng, yrng = set_up(df.index, df['GAP'], truncated = False)
    
    p = figure(width = 1000, height = 700,
               title="Sticky Forecast of NGDP" , 
               x_axis_label = 'Date', x_axis_type = 'datetime',
               y_axis_label = 'Unexpected NGDP Growth', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    
    p.line(xdata,ydata, color = 'blue', legend = 'Sticky-forecast of NGDP')
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.legend.location = 'bottom_right'
    p.ygrid.grid_line_color = None
    p.yaxis.formatter=NumeralTickFormatter(format="0.0%")

    export_png(p,'images/NGDP_gap.png')

    return p

def chart_philips_curve(df):
    xdata, ydata, xrng, yrng = set_up(df['UNRATE'], df['GAP'], 
                                      truncated = False, margins = .005)
    
    p = figure(width = 700, height = 700,
               title="Phillips Curve with Rational Expectations: United States, 1997 to 2018" , 
               x_axis_label = 'Civilian Unemployment Rate', 
               y_axis_label = 'Unexpected NGDP Growth (Sticky Forecast Approach)', 
               y_range = yrng, x_range = xrng)
    p.line(xrng,[0,0], color = 'black')
    p.line([0,0],yrng, color = 'black')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xdata, ydata)
    leg = 'R = {:.4f}, P-Value = {:.4e}, Slope = {:.4f}'.format(r_value,p_value,slope)
    p.line(xdata, xdata*slope + intercept, legend = leg, color = 'black')
    p.circle(xdata,ydata, color = 'blue',size = 2)
    
    p.xaxis[0].ticker.desired_num_ticks = 10
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.formatter=NumeralTickFormatter(format="0.0%")
    p.yaxis.formatter=NumeralTickFormatter(format="0.0%")
    
    export_png(p,'images/PhilCurve.png')
    
    return p

df = make_df()
for i in range(2,21):
    df.iloc[:,i] = df.iloc[:,i].shift(i-1) 

df['Forecast'] = df.iloc[:,1:].apply(lambda x: np.mean(x), axis = 1)
df['GAP'] = np.log(df['NGDP1']) - np.log(df['Forecast'])
df['UNRATE'] = make_u_ser()

rgdp = pd.read_csv('data/GDPC1.csv', names = ['DATE', 'RGDP']).iloc[202:,:]
ind = pd.DatetimeIndex(rgdp['DATE'])
rgdp = rgdp.set_index(ind)

df['RGDP'] = rgdp['RGDP'].astype(float)/100
df = df.dropna()

ps = [chart_NGDP_ser(df), chart_gap_ser(df), chart_philips_curve(df)]

output_file("images/stickyPCurve.html")
show(column(row(ps[0],ps[1]),ps[2]))
