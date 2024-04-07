#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:29:22 2024

@author: aniruddh

Strategy Rules:
    If today’s close is below yesterday’s five-day low, go long at the close.
    Sell at the close when the two-day RSI closes above 50.
    We have a time stop of five days if the sell criterium is not triggered.
"""

import vectorbt as vbt
#from ../data_puller import get_data_yf
import pandas as pd
import numpy as np

 
def nday_low(lows, window):
    try:
        assert(isinstance(window, int) or isinstance(window, np.int64))
    except AssertionError:
        print(type(window))
        print(window)
        return
    lows = pd.DataFrame(lows)
    nday_lows = lows.rolling(window=window).min()
    nday_lows = np.roll(nday_lows, 1, axis=0)
    nday_lows[0] = np.NaN
    
    return nday_lows


def strategy1(low, closes, low_window, rsi_window, rsi_threshold, max_dim):
    ind = vbt.IndicatorFactory(
        class_name = 'N-day Low',
        short_name = 'low_n',
        input_names = ['lows'],
        param_names = ['window'],
        output_names = ['value']
        ).from_apply_func(nday_low, window=14)
    
    nday_lows = np.array(ind.run(low, window=low_window).value)
    
    rsis  = np.array(vbt.RSI.run(closes, window=rsi_window).rsi)
    #print(np.shape(rsis), np.shape(nday_lows))
    
    entries = np.where(closes < nday_lows, 1, 0)
    
    exits = np.zeros(np.shape(entries))
    exits[-1] = -1
    
    assert(np.shape(rsis) == np.shape(closes) == np.shape(low)\
           == np.shape(nday_lows) == np.shape(entries) == np.shape(exits))
        
    num_syms = np.shape(closes)[1]
    num_pds = np.shape(closes)[0]
    
    in_market = np.zeros(num_syms)
    days_in_market = np.zeros(num_syms)
    
    for i in range(num_syms):
        for j in range(num_pds):
            if not in_market[i]:
                if entries[j][i]:
                    in_market[i] = 1
            else: # in_market
                #assert(type(dataframe['Close'].iloc[i-1]) == np.float64)
                exit_cond = rsis[j][i] > rsi_threshold or days_in_market[i] == max_dim
                if exit_cond:
                    in_market[i] = 0
                    exits[j][i] = -1
                    days_in_market[i] = 0
                else: # curr_rsi <= rsi_threshold and days_in_market < max_dim
                    days_in_market[i] += 1
    
    return entries, exits

pairs = ['EURUSD=X', 'AUDUSD=X', 'GBPUSD=X', 'CADUSD=X']
num_pairs = len(pairs)
defaults = {'low_window':5, 'rsi_window':2, 'rsi_threshold':50, 'max_dim':5}
ranges = {'low_window':[3,4,5,6,7], 'rsi_window':[2,3,4,5,6],
          'rsi_threshold':[30,40,50,60,70], 'max_dim':[3,4,5,6,7]}

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
    
    if test_type == 'range':
        ind = vbt.IndicatorFactory(
            class_name = 'Strategy 1',
            short_name = 'strat1',
            input_names = ['low', 'closes'],
            param_names = list(defaults),
            output_names = ['entries', 'exits'],
            ).from_apply_func(strategy1, **params, param_product=True)
    else:
        ind = vbt.IndicatorFactory(
            class_name = 'Strategy 1',
            short_name = 'strat1',
            input_names = ['low', 'closes'],
            param_names = list(defaults),
            output_names = ['entries', 'exits'],
            ).from_apply_func(strategy1, **params, param_product=True)
    
    res = ind.run(df['Low'], df['Close'], **params)
    pf = vbt.Portfolio.from_signals(df['Close'], res.entries, res.exits)
    
    ret = pf.total_return()
    if verbose:
        print('\n{} Strategy 1 Return\n'.format(name))  
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
    

'''    
    print('\n{} Strategy 1 Return\n'.format(name))    
    ret = pf.total_return()
    
    maxes = np.where(ret==max(ret))[0]
    num_params = len(defaults)
    param_names = list(defaults.keys())
    best = {param_names[i]:ret.index[maxes][0][i] for i in range(num_params)}
    #print(ret)
    print('\nBest Params: {}'.format(best))
    print('Best Return: {}'.format(max(ret)))
    
    resbest = ind.run(df['Low'], df['Close'], **best)
    pfbest = vbt.Portfolio.from_signals(df['Close'], resbest.entries, resbest.exits)
'''    

    
#   pf.plot().show()

#run_strat(df, '')