#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:26:16 2024

@author: aniruddh
"""

import pandas as pd
import numpy as np
import os
from run_backtests import pairs
import warnings
warnings.simplefilter("ignore", category=Warning)
import matplotlib.pyplot as plt

#strat1_eurusd_train = pd.read_csv('metrics/strat1/hourly/EURUSD=X_7_10.csv', index_col=0)
#strat1_eurusd_test = pd.read_csv('metrics/strat1/hourly/EURUSD=X_3_10.csv', index_col=0)

path_to_strats = 'vbt_strats'
train = []
test = []
metrics = []
sig_threshold = 65
prompt_each_iter = False
plot = True
short_names = ['Ret%', 'MaxDD%', 'MaxDD Dur', 'TotTrades', 'Win%', 'BestTrade%',
               'WorstTrade%', 'AvgWin%', 'AvgLos%', 'AvgWinDur', 'AvgLosDur',
               'PftFact', 'Exp', 'Sharpe', 'Calmar', 'Omega', 'Sortino']

long_names = ['Total Return [%]', 'Max Drawdown [%]', 'Max Drawdown Duration',
       'Total Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
       'Avg Winning Trade [%]', 'Avg Losing Trade [%]',
       'Avg Winning Trade Duration', 'Avg Losing Trade Duration',
       'Profit Factor', 'Expectancy', 'Sharpe Ratio', 'Calmar Ratio',
       'Omega Ratio', 'Sortino Ratio']


def time_delta_to_float(s, nancase=float('nan')):
    #print(s, type(s), type(s)==str)
    
    try:
        assert(type(s) == str)
    except AssertionError:
        print("Input {} isn't a string, returning {}".format(s, nancase))
        #print(s == float('nan'))
        return float('nan')
    
    days, _, rest = tuple(s.split())
    hours, mins, secs = tuple(map(float, rest.split(':')))
    return int(days) + (hours * 3600 + mins*60 + secs)/(24*3600)

    
def get_metrics(strat_name, pair, data_type='hourly', train_prop=(7,10)):
    assert(data_type == 'hourly' or data_type == 'daily')
    num_strats = len(os.listdir(path_to_strats))
    assert(isinstance(strat_name, str) and strat_name[:5]=='strat'\
        and int(strat_name[5:]) <= num_strats)
    
    filename_train = 'metrics/{}/{}/{}_{}_{}.csv'.format(strat_name, data_type,
                                               pair[0]+pair[1]+'=X', *train_prop)
    
    test_prop = (train_prop[1] - train_prop[0], train_prop[1])
    
    filename_test = 'metrics/{}/{}/{}_{}_{}.csv'.format(strat_name, data_type,
                                               pair[0]+pair[1]+'=X', *test_prop)
    
    global train
    train = pd.read_csv(filename_train, index_col=0)
    global test
    test = pd.read_csv(filename_test, index_col=0)
    assert(np.all(train.index == test.index))
    
    return train, test


def get_correlations(strat_name, pair, data_type='hourly', train_prop=(7,10)):
    train, test = get_metrics(strat_name, pair, data_type, train_prop)
    
    global metrics
    metrics = list(train.index)[5:]
    result = pd.Series(index=metrics)
    
    for metric in metrics:
        train_metric = train.loc[metric]
        test_metric = test.loc[metric]
        if metric.split()[-1] == 'Duration':
            train_metric = pd.Series(map(time_delta_to_float, train_metric))
            test_metric = pd.Series(map(time_delta_to_float, test_metric))
        
        result[metric] = train_metric.corr(test_metric)
        
    #print(type(result))
    return result


def display(df):
    for metric in df.index:
        ser = df.loc[metric]
        out = '{}\tmin:{:.4f}\tmax:{:.4f}\tvar:{:.4f}'.format(metric, min(ser), max(ser), ser.var())
        print(out)

# ROBUSTNESS
'''
result = pd.DataFrame({'Train Avg':[0 for i in long_names], 
                       'Test Avg':[0 for j in long_names],
                       'n':[0 for k in long_names]},
                   index=long_names)
slash = lambda p: '{}/{}'.format(*p)
rets = pd.DataFrame({'Train Return':[0 for i in pairs], 
                       'Test Return':[0 for j in pairs]},
                   index=map(slash, pairs))
for pair in pairs:
    train, test = get_metrics('strat2', pair)
    train_def, test_def = train['(5, 35, 50, 200)'], test['(5, 35, 50, 200)']
    #print('pair = {}'.format(pair))
    for metric in long_names:
        train_metric, test_metric = train_def.loc[metric], test_def.loc[metric]
        invalid = [float('nan'), np.inf, np.nan, float('inf'), 'nan', 'inf']
        isinvalid = lambda x: x in invalid
        
        if isinvalid(train_metric) or isinvalid(test_metric):
            continue
        elif metric.split()[-1] == 'Duration':
            #print(metric)
            result['Train Avg'].loc[metric] += time_delta_to_float(train_metric)
            result['Test Avg'].loc[metric] += time_delta_to_float(test_metric)
        else:
            #print(train_metric, test_metric, isinvalid(train_metric), isinvalid(test_metric))
            result['Train Avg'].loc[metric] += float(train_metric)
            result['Test Avg'].loc[metric] += float(test_metric)
        result['n'].loc[metric] += 1
    
    ret = 'Total Return [%]'
    best = train.columns[np.argmax(train.loc[ret].map(float))]
    train_ret, test_ret = train[best].loc[ret], test[best].loc[ret]
    rets['Train Return'].loc[slash(pair)] += float(train_ret)
    rets['Test Return'].loc[slash(pair)] += float(test_ret)

result.index = short_names
result['Train Avg'] = result['Train Avg']/result['n']
result['Test Avg'] = result['Test Avg']/result['n']
result['Difference'] = result['Train Avg'] - result['Test Avg']
result = result.drop('n', axis=1)
print(result)

#rets['Difference'] = rets['Train Return'] - rets['Test Return']
#print(rets)
#print(rets['Difference'].mean(), rets['Difference'].var())
'''

# STABILITY

df = pd.DataFrame(columns=pairs)
for pair in pairs:
    correlations = get_correlations('strat3', pair)
    df[pair] = correlations
    
df = df.loc[long_names]
#df = df.drop('Total Closed Trades', axis=0)

df.index = short_names

notable_metrics = set(long_names)
for i, pair in enumerate(pairs):
    title = '{}/{} IS/OOS Metric Correlations'.format(*pair)
    corrs = df[pair]*100
    out = ['Notably High Correlations for {}:'.format(pair)]
    notable_corrs = []
    for j, metric in enumerate(corrs.index):
        c = 0 if np.isnan(corrs[metric]) else int(corrs[metric])
        if c >= sig_threshold:
            notable_corrs.append(long_names[j])
            out.append(long_names[j] + ': {}%'.format(c))
    
    if plot:
        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
        ax = axs
        corrs.plot(kind='bar', title=title, ax=ax, fontsize=10, rot=45)
        for p in ax.patches:
            ht = '{}'.format(int(p.get_height()))
            ax.annotate(ht, (p.get_x(), max(p.get_height() * 1.005, 0.1)))
        
    print(*out, sep='\n\t- ', end='\n\n')
    notable_metrics = notable_metrics.intersection(set(notable_corrs))
    if prompt_each_iter:
        action = input('Continue? ("n" to cancel): ')
        if len(action) >= 1 and action[0].lower() == 'n':
            break

print('Notable Metrics:\n\t- ', end='')
print(*notable_metrics, sep='\n\t- ', end='\n\n')
display(df)


