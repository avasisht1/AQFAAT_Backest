#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 22:36:35 2024

@author: aniruddh
"""

"""
This file is supposed to be an interface between the AlphaVantage API and my
code. I'll be trying to use the API to load data into the program, which can
then be import into other program files.
"""

import requests
import pandas as pd
import os
import yfinance as yf


# Get 1-hour OHLC data (default, customizable)
def get_data_av(currency_pair, size, format_for='nautilus', data_type='FX_INTRADAY', 
                 interval='60min', apikey='CV4O3KUIMS9TVCLR'):
    
    path = 'Data/AlphaVantage/'
    # First check if we already have data on that currency pair
    filename = path+'{}_{}_{}_{}.csv'.format(*currency_pair, size, interval)
    if not os.path.exists(filename):
        print("Data doesn't exist in {}".format(path))
        print("Querying AlphaVantage API ...")
        function = 'function={}'.format(data_type)
        from_currency = 'from_symbol={}'.format(currency_pair[0])
        to_currency = 'to_symbol={}'.format(currency_pair[1])
        outputsize = 'outputsize={}'.format(size)
        data_name = interval
        
        url = 'https://www.alphavantage.co/query?{}&{}&{}'\
            .format(function, from_currency, to_currency)
        if data_type == 'FX_INTRADAY':
            url += '&interval={}&{}&apikey={}'.format(interval, outputsize, apikey)
        elif data_type == 'FX_DAILY':
            url += '&{}&apikey={}'.format(outputsize, apikey)
            data_name = 'Daily'
        elif data_type == 'FX_WEEKLY' or data_type == 'FX_MONTHLY':
            url += 'apikey={}'.format(apikey)
            data_name = 'Weekly'
        elif data_type == 'FX_MONTHLY':
            url += 'apikey={}'.format(apikey)
            data_name = 'Monthly'
        else:
            print("Failure; data_type invalid")
            return
            
        r = requests.get(url)
        data = r.json()
        #return data
        print("Formatting data into pd.DataFrame")
        df = pd.DataFrame(data['Time Series FX ({})'.format(data_name)]).transpose()
        df.columns = ['Open', 'High', 'Low', 'Close']
        df.to_csv(filename)
        print("Done, data saved as {}".format(filename))
    
    print("Loading data from {}\n".format(filename))
    df = pd.read_csv(filename)
    
    if format_for == 'nautilus':
        df.columns = ["timestamp", "open", "high", "low", "close"]
        df.index = pd.core.indexes.datetimes.DatetimeIndex(df['timestamp'])
        df = df.drop('timestamp', axis=1)
        
    elif format_for == 'manual':
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        df.index = pd.core.indexes.datetimes.DatetimeIndex(df['Timestamp'])
        df = df.drop('Timestamp', axis=1)
    
    df.sort_index(inplace=True)
    
    return df


def get_data_yf(tickers, interval, period, *args, format_for='nautilus'):
    #maxes = {'1m':'7d', '60m':'2y', '1h':'2y', '2m':'60d', '5m':'60d',
    #         '15m':'60d', '30m':'60d', '90m':'60d'}
    path = 'Data/YahooFinance/'
    if len(args) > 2:
        print("Too many arguments; args 4 and 5 must be start and end")
        return
    elif len(args) == 1:
        print("Too few arguments; args 4 and 5 must be start and end")
        return
    c1,c2 = tickers[:3], tickers[3:6]
    filename = path + '{}_{}_{}_{}.csv'.format(c1,c2,interval,period)
    if not os.path.exists(filename):
        print("Data doesn't exist in {}".format(path))
        print("Querying YahooFinance ...")
        df = yf.download(tickers=tickers, interval=interval, period=period, *args)
        df.to_csv(filename)
        print("Done, data saved as {}".format(filename))
        
    print("Loading data from {}\n".format(filename))
    df = pd.read_csv(filename)
    df = df.drop(['Volume', 'Adj Close'], axis=1)
    
    if format_for == 'nautilus':
        df.columns = ["timestamp", "open", "high", "low", "close"]
        df.index = pd.core.indexes.datetimes.DatetimeIndex(df['timestamp'])
        df = df.drop('timestamp', axis=1)
        
    elif format_for == 'manual':
        df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close']
        df.index = pd.core.indexes.datetimes.DatetimeIndex(df['Timestamp'])
        df = df.drop('Timestamp', axis=1)
    
    df.sort_index(inplace=True)
    
    return df

#euro_dollar_compact_1d = get_data_av(('EUR','USD'), "compact", 'FX_DAILY', '1d')
#euro_dollar_full_1h = get_data_av(currency_pair=('EUR','USD'), size="full")

#euro_dollar_yf_1h = yf.download(tickers='EURUSD=X', interval='1h', period='2y')
#euro_dollar_yf_1d = get_data_yf(tickers='EURUSD=X', interval='1d', period='100d')
