#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:36:32 2024

@author: aniruddh

Strategy Rules:
    Enter at the close when the S&P 500 closes below a five-day RSI of 35.
    Sell at the close when the same five-day RSI crosses above 50.
    Optional: Also require that close > 200-day SMA to enter
"""

import vectorbt as vbt
#from data_puller import get_data_yf
import pandas as pd
import numpy as np


def strategy2(closes, rsi_window, rsi_lower, rsi_upper, sma_window):
    rsis  = np.array(vbt.RSI.run(closes, window=rsi_window).rsi)
    smas  = np.array(vbt.MA.run(closes, window=sma_window).ma) if sma_window!=-1 else None
    if sma_window == -1:
        entries = np.where(rsis < rsi_lower, 1, 0)
    else:
        entries = np.where((rsis < rsi_lower) & (smas < closes), 1, 0)
        
    exits = np.where(rsis > rsi_upper, -1, 0)
    exits[-1] = -1
    
    return entries, exits

    
pairs = ['EURUSD=X', 'AUDUSD=X', 'GBPUSD=X', 'CADUSD=X']
num_pairs = len(pairs)
defaults = {'rsi_window':5, 'rsi_lower':35,
            'rsi_upper':50, 'sma_window':200}

ranges = {'rsi_window':[3,4,5,6,7], 'rsi_lower':[25,35,45],
          'rsi_upper':[50,60,70], 'sma_window':[150,175,200,225,250]}

df = vbt.YFData.download(pairs[0], missing_index='drop').get()

pf = 0
total_returns = 0
max_returns = 0
best_params = 0
max_returns = 0

def run_strategy(df, name, test_type='range', params=ranges, verbose=False):
    is_set = lambda v: (isinstance(v, int) or (isinstance(v, list) and len(v)==1))
    try:
        assert((test_type == 'range' and not is_set(params)) 
               or (test_type == 'set' and is_set(params)))
    except AssertionError:
        print('Assertion Failed: Params == {}, test_type == {}'.format(params, test_type))
        raise AssertionError
        
    ind = vbt.IndicatorFactory(
        class_name = 'Strategy 2',
        short_name = 'strat2',
        input_names = ['closes'],
        param_names = list(defaults),
        output_names = ['entries', 'exits'],
        ).from_apply_func(strategy2, **params, param_product=True)
    '''     
    else:
        ind = vbt.IndicatorFactory(
            class_name = 'Strategy 2',
            short_name = 'strat2',
            input_names = ['closes'],
            param_names = list(defaults),
            output_names = ['entries', 'exits'],
            ).from_apply_func(strategy2, **params, param_product=True)
    '''
    res = ind.run(df['Close'], **params)
    pf = vbt.Portfolio.from_signals(df['Close'], res.entries, res.exits)
        
    #print(pf[(3,25,50,150)].stats(settings=dict(freq='d')))
        
    ret = pf.total_return()
    if verbose:
        print('\n{} Strategy 2 Return\n'.format(name))  
        print(ret)
    
    if test_type == 'set':
        maxes = np.where(ret==max(ret))[0]
        num_params = len(defaults)
        param_names = list(defaults.keys())
        best = {param_names[i]:ret.index[maxes][0][i] for i in range(num_params)}
        #print(ret1)
        if verbose:
            print('\nBest Params: {}'.format(best))
            print('Best Return: {}'.format(max(ret)))
        return pf, best
    # else
    global max_returns
    return pf, pf.total_return()