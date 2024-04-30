#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 06:33:12 2024

@author: aniruddh
"""
import pandas as pd
import matplotlib.pyplot as plt


plot = True
read = True
src = 'log1.csv'

if not read:
    lines = []
    with open('logs/Trade.log') as reader:
        lines = reader.readlines()
    
    useful_lines = []
    for line in lines:
        if "'Id'" in line:
            start = 0
            end = 0
            for i, char in enumerate(line):
                if char=='{':
                    start = i
                elif char=='}':
                    end = i+1
            useful_lines.append(line[start:end])#.split('\':'))
    
    df = pd.DataFrame(columns=['Id', 'wanted_price', 'actual_price',
                               'wanted_time', 'actual_time',
                               'pair', 'units'], 
                      index=[i for i in range(len(useful_lines))])
    
    for i, line in enumerate(useful_lines):
        d = eval(line)
        #print(d, type(d))
        for k in df.columns:
            df.loc[i, k] = d[k]
    
    df['actual_time'] = df['actual_time'].apply(pd.to_datetime)
    df['wanted_time'] = df['wanted_time'].apply(pd.to_datetime)
    df['latency'] = (df['actual_time'] - df['wanted_time']).apply(pd.Timedelta.total_seconds)
    df['slippage'] = df['actual_price'] - df['wanted_price']
    metrics = df[['pair', 'units', 'slippage', 'latency']].copy()

else:
    df = pd.read_csv(src)
    df = df.drop(['Unnamed: 0'], axis=1)
    df['actual_time'] = df['actual_time'].apply(pd.to_datetime)
    df['wanted_time'] = df['wanted_time'].apply(pd.to_datetime)
    df['latency'] = (df['actual_time'] - df['wanted_time']).apply(pd.Timedelta.total_seconds)
    df['slippage'] = df['actual_price'] - df['wanted_price']

#print(len(df.index))

if plot:
    x1=list(df['wanted_time'])
    x2=list(df['actual_time'])
    y1=list(df['wanted_price'])
    y2=list(df['actual_price'])
    
    plt.scatter(x1,y1, label='Wanted Times/Prices')
    plt.scatter(x2,y2, label='Actual Times/Prices')
    plt.title('Time of Trade vs Price Filled')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()