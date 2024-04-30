#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 00:02:49 2024

@author: aniruddh
"""

import requests
import pandas as pd
import login_details as defs
import os

path = os.getcwd()
parent = os.path.abspath(os.path.join(path, os.pardir))

pairs = ['EUR_USD', 'AUD_USD', 'GBP_USD', 'USD_CAD',
         'USD_JPY', 'USD_CHF', 'USD_HKD', 'EUR_GBP']

def fetch_candles(pair_name, count, granularity):
    session = requests.Session()
    url = '{}/instruments/{}/candles'.format(defs.OANDA_URL, pair_name)
    
    params = dict(
        count = count,
        granularity = granularity, 
        price = 'MBA'
    )
    
    response = session.get(url, params=params, headers=defs.SECURE_HEADER)
    session.close()
    return response.status_code, response.json()
    
    
def get_candles(json_response):
    
    our_data = []
    for candle in json_response['candles']:
        if not candle['complete']:
            continue
        new_dict = dict(
            time = candle['time'],
            volume = candle['volume']
        )
        for price in ['bid', 'mid', 'ask']:
            for oh in list('ohlc'):
                new_dict['{}_{}'.format(price, oh)] = candle[price][oh]
        
        our_data.append(new_dict)
    
    return pd.DataFrame.from_dict(our_data)


def save_data(candles_df, pair, granularity):
    candles_df.to_pickle('{}/Data/Oanda/{}_{}.pkl'.format(parent, pair, granularity))
    

def create_data(pair, count, granularity):
    code, json_data = fetch_candles(pair, count, granularity)
    if code != 200:
        print('Error {} occurred trying to get pair {}'.format(code, pair))
        return
    
    df = get_candles(json_data)
    print('{} loaded {} candles from {} to {}'.format(pair, df.shape[0],
                                                      df.time.min(), 
                                                      df.time.max()))
    save_data(df, pair, granularity)

for pair in pairs:
    create_data(pair, 4000, 'M1')