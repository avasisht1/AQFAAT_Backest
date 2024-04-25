#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:34:59 2024

@author: aniruddh

Strategy Rules:
    Calculate an average of the H-L over the last 25 days.
    Calculate the (C-L)/(H-L) ratio every day (IBS).
    Calculate a band 2 times lower than the high over the last 25 days by using the average from point number 1.
    If the close is under the band in number 3, and point 2 (IBS) has a higher value than 0.4, then go long at the close.
    Exit when the close is higher than yesterday’s high.
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


def strategy3(low, high, closes, range_window, high_window, band_width, ibs_threshold):
    assert(isinstance(range_window, int) or isinstance(range_window, np.int64))
    assert(isinstance(high_window, int) or isinstance(high_window, np.int64))
    assert(isinstance(band_width, int) or isinstance(band_width, np.int64))
    assert(isinstance(ibs_threshold, float))
    
    ibs = (closes-low)/(high-low)
    average_range = vbt.IndicatorFactory(
        class_name = 'Average Range',
        short_name = 'avg_range',
        input_names = ['lows', 'highs'],
        param_names = ['window'],
        output_names = ['avg_range']
        ).from_apply_func(avg_range, window=range_window)
    
    average_high = vbt.IndicatorFactory(
        class_name = 'Average High',
        short_name = 'avg_high',
        input_names = ['high'],
        param_names = ['window'],
        output_names = ['avg_high']
        ).from_apply_func(avg_high, window=high_window)
    
    avg_ranges = np.array(average_range.run(low, high, window=range_window).avg_range)
    avg_highs = np.array(average_high.run(high, window=high_window).avg_high)
    band = avg_highs - (band_width * avg_ranges)
    assert(isinstance(avg_ranges, np.ndarray))
    assert(isinstance(avg_highs, np.ndarray))
    assert(isinstance(band, np.ndarray))
    try:
        assert(np.shape(avg_ranges) == np.shape(avg_highs) == np.shape(band) == (np.shape(low)[0], 1))
    except AssertionError:
        print('avg_ranges: {}'.format(np.shape(avg_ranges)))
        print('avg_highs: {}'.format(np.shape(avg_highs)))
        print('band: {}'.format(np.shape(band)))
        print('np.shape(low)[0]: {}'.format(np.shape(low)[0]))
        print(band_width*avg_ranges)
        print(avg_highs)
        print(band)
        return
    prev_highs = np.roll(high, 1)
    prev_highs[0] = np.nan
    
    #print(band)
    
    entries = np.where((closes < band) & (ibs < ibs_threshold), 1, 0)
    exits = np.where(closes < prev_highs, -1, 0)
    exits[-1] = -1
    #print(np.shape(entries), np.shape(exits))
    
    return entries, exits


pairs = ['EURUSD=X', 'AUDUSD=X', 'GBPUSD=X', 'CADUSD=X']
pairs = 'EURUSD=X'

num_pairs = len(pairs)
defaults = {'range_window':25, 'high_window':25,
          'band_width':2, 'ibs_threshold':0.4}

ranges = {'range_window':[15,20,25,30,35], 'high_window':[15,20,25,30,35],
          'band_width':[1,2,3,4,5], 'ibs_threshold':[0.3,0.4,0.5,0.6]}

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
        class_name = 'Strategy 3',
        short_name = 'strat3',
        input_names = ['low', 'high', 'closes'],
        param_names = list(defaults),
        output_names = ['entries', 'exits'],
        ).from_apply_func(strategy3, **params)
        
    res = ind.run(df['Low'], df['High'], df['Close'], **params, param_product=True)
    pf = vbt.Portfolio.from_signals(df['Close'], res.entries, res.exits)
    
    ret = pf.total_return()
    if verbose:
        print('\n{} Strategy 3 Return\n'.format(name))  
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


#pf, ret = run_strategy(df, '', test_type='set', params=defaults)