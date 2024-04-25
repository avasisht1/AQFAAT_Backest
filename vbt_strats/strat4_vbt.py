#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:06:04 2024

@author: aniruddh

Strategy Rules:
    If today’s IBS (C-L)/(H-L) < 0.9, short at the close
    Close position (ie buy to cover at the close) when IBS ≤ 0.3
"""

import vectorbt as vbt
#from data_puller import get_data_yf
import pandas as pd
import numpy as np


def avg_range(lows, highs, window):
    assert(isinstance(window, int) or isinstance(window, np.int64))
    rng = pd.DataFrame(highs-lows)
    avg_diffs = rng.rolling(window=window).mean()
    assert(np.shape(avg_diffs) == np.shape(lows) == np.shape(highs))
    
    return avg_diffs


def avg_high(high, window):
    assert(isinstance(window, int) or isinstance(window, np.int64))
    high = pd.DataFrame(high)
    avg_highs = high.rolling(window=window).mean()
    assert(np.shape(avg_highs) == np.shape(high))
    
    return avg_highs


def strategy4(low, high, closes, ibs_upper, ibs_lower):
    try:
        assert(isinstance(ibs_upper, float) or isinstance(ibs_upper, np.float64))
    except AssertionError:
        print('ibs_upper is a bad type ({})'.format(type(ibs_upper)))
        return
    try:
        assert(0 <= ibs_upper <= 1)
    except AssertionError:
        print('ibs_upper has a bad value ({})'.format(ibs_upper))
        return
        
    try:
        assert(isinstance(ibs_lower, float) or isinstance(ibs_lower, np.float64))
    except AssertionError:
        print('ibs_lower is a bad type ({})'.format(type(ibs_lower)))
        return
    try:
        assert(0 <= ibs_lower <= 1)
    except AssertionError:
        print('ibs_upper has a bad value ({})'.format(ibs_lower))
        return

    
    ibs = (closes-low)/(high-low)
    
    try:
        assert(ibs.shape == closes.shape == low.shape == high.shape)
    except AssertionError:
        print('ibs vector is shaped badly')
        print('np.shape(ibs): {}'.format(np.shape(ibs)))
        print('np.shape(low): {}'.format(np.shape(low)))
        print('np.shape(high): {}'.format(np.shape(high)))
        print('np.shape(close): {}'.format(np.shape(closes)))
        return
    
    entries = np.where(ibs < ibs_upper, 1, 0)
    exits = np.where(ibs <= ibs_lower, -1, 0)
    exits[-1] = -1
    
    return entries, exits


pairs = ['EURUSD=X', 'AUDUSD=X', 'GBPUSD=X', 'CADUSD=X']
pairs = 'EURUSD=X'

num_pairs = len(pairs)
defaults = {'ibs_upper': 0.9, 'ibs_lower': 0.3}
ranges = {'ibs_upper':[0.5, 0.6, 0.7, 0.8, 0.9], 'ibs_lower':[0.1, 0.2, 0.3, 0.4, 0.5]}

df = vbt.YFData.download(pairs[0], missing_index='drop').get()
sames = np.where(df['High'] == df['Low'])[0]
df.drop([df.index[i] for i in sames], axis=0, inplace=True)

pf = 0
total_returns = 0
max_returns = 0
best_params = 0
max_returns = 0

def run_strategy(df, name, test_type='range', params=ranges, verbose=False):
    numeric = (int, float, np.int64, np.float64)
    is_set = lambda v: (isinstance(v, numeric) or (isinstance(v, list) and len(v)==1))
    try:
        assert((test_type == 'range' and not np.all( [is_set(params[k]) for k in params] )) 
               or (test_type == 'set' and np.all( [is_set(params[k]) for k in params] ) ))
    except AssertionError:
        print('Assertion Failed: Params == {}, test_type == {}'.format(params, test_type))
        raise AssertionError
    
    ind = vbt.IndicatorFactory(
        class_name = 'Strategy 4',
        short_name = 'strat4',
        input_names = ['low', 'high', 'closes'],
        param_names = list(defaults),
        output_names = ['entries', 'exits'],
        ).from_apply_func(strategy4, **params)
        
    res = ind.run(df['Low'], df['High'], df['Close'], **params, param_product=True)
    pf = vbt.Portfolio.from_signals(df['Close'], 
                                    short_entries=res.entries,
                                    short_exits=res.exits)
    
    ret = pf.total_return()
    if verbose:
        print('\n{} Strategy 4 Return\n'.format(name))  
        print(ret)
    
    if test_type == 'range':
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


#pf, ret = run_strategy(df, '', test_type='range', params=ranges, verbose=True)


