#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:07:11 2024

@author: aniruddh
"""

#from data_puller import get_data_av, get_data_yf, format_manual, format_nautilus
#from opt_universe import optimize_strat, create_table, plot_search_space
#from backtest import run_test, hourly_to_daily, plot_equity_line, apply_strat
import pandas as pd
import numpy as np
import vectorbt as vbt
import importlib.util
import os
import itertools

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

EUR = 'EUR'
JPY = 'JPY'
CNY = 'CNY'
GBP = 'GBP'
AUD = 'AUD'
CAD = 'CAD'
USD = 'USD'
CHF = 'CHF'
HKD = 'HKD'
KRW = 'KRW'

pairs = [(EUR, USD), (AUD, USD), (GBP, USD), (USD, CAD), (USD, JPY),
         (USD, CNY), (USD, CHF), (USD, HKD), (EUR, GBP), (USD, KRW)]

'''
strats = [strategy1, strategy2, strategy3]

defaults1 = {'low_window':[3,4,5,6,7],
             'rsi_window':[2,3,4,5,6], 
             'rsi_threshold':[30,40,50,60,70],
             'max_dim':[3,4,5,6,7]}

defaults2 = {'rsi_window':[3,4,5,6,7],
             'rsi_lower':[25,35,45],
             'rsi_upper':[50,60,70],
             'sma_window':[150,175,200,225,250]}

defaults3 = {'range_window':[15,20,25,30,35],
             'high_window':[15,20,25,30,35],
             'band_width':[1,2,3,4,5],
             'ibs_threshold':[0.3,0.4,0.5,0.6]}

def hourly_test(pair, benchmark=None, plot_equity=False):
    df = get_data_av(pair, 'compact', format_for='manual')
    run_test(df, '{}/{} Low5_RSI2 (Daily Data)'.format(*pair), benchmark=benchmark,
             plot_equity=plot_equity)
    
    
def daily_test(pair, benchmark=None, plot_equity=False):
    df = get_data_av(pair, 'full', format_for='manual')
    df_daily = hourly_to_daily(df)
    run_test(df_daily, '{}/{} Low5_RSI2 (Hourly Data)'.format(*pair), benchmark=benchmark,
             plot_equity=plot_equity)
    return df_daily

opt_universe = {"low_pd":[3,10,1], "rsi_pd":[2,8,1], "rsi_threshold":[30,71,10], "max_dim":[3,8,1]}

snp = pd.read_csv('Data/YahooFinance/SPX.csv').drop(['Adj Close', 'Volume'], axis=1)
snp = format_manual(snp)
benchmark=(snp, 'Buy&Hold S&P 500', True)
best_params = {pair:[] for pair in pairs}
'''
train_prop = (7,10)
num_pairs = len(pairs)

folder_path = 'vbt_strats'

for pair in pairs[5:]:
    #default = (5, 2, 50, 5)
    # Get the actual data for the pair
    #pair = pairs[i]
    name = pair[0]+pair[1]+'=X'
    # Change the following to reading from csv after it's all been stored
    df = vbt.YFData.download(name, missing_index='drop').get()
    
    split_point = len(df.index)*train_prop[0]//train_prop[1]
    df_train = df.iloc[:split_point]
    df_test = df.iloc[split_point:]
    
    sames_train = np.where(df_train['High'] == df_train['Low'])[0]
    df_train.drop([df_train.index[i] for i in sames_train], axis=0, inplace=True)
    
    sames_test = np.where(df_test['High'] == df_test['Low'])[0]
    df_test.drop([df_test.index[i] for i in sames_test], axis=0, inplace=True)
    
    # Remove later after all data has been stored
    df.to_csv('Data/vbt/{}.csv'.format(name))
    print('Historical data written to Data/vbt/{}.csv'.format(name))
    
    # Get functions that run the strategies
    # Iterate over the files and import functions from each file
    for file in os.listdir(folder_path):
        metrics_train = pd.DataFrame()
        metrics_test = pd.DataFrame()
        if not file.endswith('.py'):
            continue
        strat_number = int(file.split('_')[0][5:])
        print('Getting metrics for Strategy {}'.format(strat_number))
        # Construct the module name from the file name
        module_name = file[:-3]  # Remove the .py extension
        
        print('Trying to import strategy')
        
        strat = importlib.import_module(module_name)
                
        # Run unoptimized strategy on train and test set
        pf_train, _ = strat.run_strategy(df_train, name, test_type='range')
        pf_test, _ = strat.run_strategy(df_test, name, test_type='range')
        
        print('Compiling metrics ...')
        for combo in itertools.product(*[strat.ranges[param] for param in strat.ranges]):
            train_metrics = pf_train[combo].stats(settings=dict(freq='d'), silence_warnings=True)
            metrics_train.index = train_metrics.index
            metrics_train = pd.concat([metrics_train, train_metrics], axis=1)
        
            test_metrics = pf_test[combo].stats(settings=dict(freq='d'), silence_warnings=True)
            metrics_test.index = test_metrics.index
            metrics_test = pd.concat([metrics_test, test_metrics], axis=1)
        
        filename_train = 'metrics/strat{}/{}_{}_{}.csv'.format(strat_number, name, *train_prop)
        print('Done! Writing training metrics to {}'.format(filename_train))
        metrics_train.to_csv(filename_train)
        print('Train metrics written to {}'.format(filename_train))
        
        test_prop = (train_prop[1]-train_prop[0], train_prop[1])
        filename_test = 'metrics/strat{}/{}_{}_{}.csv'.format(strat_number, name, *test_prop)
        
        print('Writing testing metrics to {}'.format(filename_test))
        metrics_test.to_csv(filename_test)
        print('Test metrics written to {}\n'.format(filename_test))
        
        #action = input('Continue to next strategy? ("n" to quit): ')
        #if len(action) > 0 and action[0].lower() == 'n':
        #    break
        
    #if i < num_pairs-1:
    #    action = input('Continue to {}? ("n" to quit): '.format(pairs[i+1]))
    #    if len(action)>0 and action[0].lower() == 'n':
    #        break
    

    # Run the tests and plot equity lines with and without the benchmark comparison
'''    
        s1 = vbt.IndicatorFactory(
            class_name = 'Strategy 1',
            short_name = 'strat1',
            input_names = ['low', 'closes'],
            param_names = list(defaults1),
            output_names = ['entries', 'exits'],
            ).from_apply_func(strategy1, **defaults1)
        
        s2 = vbt.IndicatorFactory(
            class_name = 'Strategy 2',
            short_name = 'strat2',
            input_names = ['closes'],
            param_names = list(defaults2),
            output_names = ['entries', 'exits'],
            ).from_apply_func(strategy2, **defaults2)
        
        s3 = vbt.IndicatorFactory(
            class_name = 'Strategy 3',
            short_name = 'strat3',
            input_names = ['low', 'high', 'closes'],
            param_names = list(defaults3),
            output_names = ['entries', 'exits'],
            ).from_apply_func(strategy3, **defaults3)
    
            
        res1train = s1.run(df_train['Low'], df_train['Close'],
                           **defaults1,
                           param_product=True)
        
        res2train = s2.run(df_train['Close'],
                           **defaults2,
                           param_product=True)
        
        res3train = s3.run(df_train['Low'], df_train['High'], df_train['Close'],
                           **defaults3,
                           param_product=True)
    
        pf1train = vbt.Portfolio.from_signals(df_train['Close'], res1train.entries, res1train.exits)
        pf2train = vbt.Portfolio.from_signals(df_train['Close'], res2train.entries, res2train.exits)
        pf3train = vbt.Portfolio.from_signals(df_train['Close'], res3train.entries, res3train.exits)

        print('\n{} Strategy 1 Return\n'.format(name))    
        ret1 = pf1train.total_return()
        maxes1 = np.where(ret1==max(ret1))[0]
        num_params1 = len(defaults1)
        param_names1 = list(defaults1.keys())
        best1 = {param_names1[i]:ret1.index[maxes1][0][i] for i in range(num_params1)}
        #print(ret1)
        print('\nBest Params: {}'.format(best1))
        print('Best Return: {}'.format(max(ret1)))
        
        print('\n{} Strategy 2 Return\n'.format(name))    
        ret2 = pf2train.total_return()
        maxes2 = np.where(ret2==max(ret2))[0]
        num_params2 = len(defaults2)
        param_names2 = list(defaults2.keys())
        best2 = {param_names2[i]:ret2.index[maxes2][0][i] for i in range(num_params2)}
        #print(ret2)
        print('\nBest Params: {}'.format(best2))
        print('Best Return: {}'.format(max(ret2)))
        
        print('\n{} Strategy 3 Return\n'.format(name))    
        ret3 = pf3train.total_return()
        maxes3 = np.where(ret3==max(ret3))[0]
        num_params3 = len(defaults3)
        param_names3 = list(defaults3.keys())
        best3 = {param_names3[i]:ret3.index[maxes3][0][i] for i in range(num_params3)}
        #print(ret3)
        print('\nBest Params: {}'.format(best3))
        print('Best Return: {}'.format(max(ret3)))
'''        
'''        
        res1best = s1.run(df_train['Low'], df_train['Close'], **best1)
        res2best = s2.run(df_train['Close'], **best2)
        res3best = s3.run(df_train['Low'], df_train['High'], df_train['Close'], **best3)
        
        pf1best = vbt.Portfolio.from_signals(df_train['Close'], res1best.entries, res1best.exits)
        pf2best = vbt.Portfolio.from_signals(df_train['Close'], res2best.entries, res2best.exits)
        pf3best = vbt.Portfolio.from_signals(df_train['Close'], res3best.entries, res3best.exits)
        
        print(pf1best.stats(settings=dict(freq='d')))
        print(pf2best.stats(settings=dict(freq='d')))
        print(pf3best.stats(settings=dict(freq='d')))
        
        pf1best.plot().show()
        pf2best.plot().show()
        pf3best.plot().show()
        
        res1test = s1.run(df_test['Low'], df_test['Close'], **best1)
        res2test = s2.run(df_test['Close'], **best2)
        res3test = s3.run(df_test['Low'], df_test['High'], df_test['Close'], **best3)
    
        pf1test = vbt.Portfolio.from_signals(df_test['Close'], res1test.entries, res1test.exits)
        pf2test = vbt.Portfolio.from_signals(df_test['Close'], res2test.entries, res2test.exits)
        pf3test = vbt.Portfolio.from_signals(df_test['Close'], res3test.entries, res3test.exits)
        
        print(pf1test.stats(settings=dict(freq='d')))
        print(pf2test.stats(settings=dict(freq='d')))
        print(pf3test.stats(settings=dict(freq='d')))
        
        pf1test.plot().show()
        pf2test.plot().show()
        pf3test.plot().show()
        
        print('\n{} Strategy 1 Optimized Return\n'.format(name))    
        ret1opt = pf1test.total_return()
        print(ret1opt)
        
        print('\n{} Strategy 2 Optimized Return\n'.format(name))    
        ret2opt = pf2test.total_return()
        print(ret2opt)
        
        print('\n{} Strategy 3 Optimized Return\n'.format(name))    
        ret3opt = pf3test.total_return()
        print(ret3opt)
'''
    
    
'''    
    run_test(df_train, '{}/{} Low5_RSI2'.format(*pair), benchmark=None, plot_equity=True)
    df_train, metrics = run_test(df_train, '{}/{} Low5_RSI2'.format(*pair), benchmark=benchmark, plot_equity=True)
    f = open('{}{}_train_metrics.csv'.format(*pair), 'w')
    f.write(str(metrics))
    f.close()
    
    run_test(df_test, '{}/{} Low5_RSI2'.format(*pair), benchmark=None, plot_equity=True)
    df_test, metrics = run_test(df_test, '{}/{} Low5_RSI2'.format(*pair), benchmark=benchmark, plot_equity=True)
    f = open('{}{}_test_metrics.csv'.format(*pair), 'w')
    f.write(str(metrics))
    f.close()
    
    df_daily = daily_test(pair, benchmark=benchmark, plot_equity=True)
    
    # Now optimize the parameters over opt_universe
    #values = create_table(opt_universe)
    #values, best_values = optimize_strat(df_train, values)
    # Saving the values on the first run so I can just load them up for subsequent runs
    #values.to_csv('{}{}_values.csv'.format(*pair))
    values = pd.read_csv('GBPUSD_values.csv')
    #if len(best_values)>1:
    #    print('Multiple Optima found, using the first one')
    
    #x = best_values[0]
    #vals = values.iloc[x]
    #low_pd, rsi_pd, rsi_threshold, max_dim = int(vals.low_pd), int(vals.rsi_pd), \
    #                                            int(vals.rsi_threshold), int(vals.max_dim)
                                                
    best1 = plot_search_space(df_train, values, 'low_pd', 'rsi_pd', ('rsi_threshold', 50), ('max_dim', 5))
    best2 = plot_search_space(df_train, values, 'rsi_threshold', 'max_dim', ('low_pd', best1[0]), ('rsi_pd', best1[1]))
    best = *best1, *best2
    #(low_pd, rsi_pd, rsi_threshold, max_dim)
    best_params[pair] = best
    
    apply_strat(df_train, 1000, *best, keep_cols=False)
    plot_equity_line(df_train, name='{}/{} (Best params)'.format(*pair))
    
    apply_strat(df_test, 1000, *best, keep_cols=False)
    plot_equity_line(df_test, name='{}/{} (Best params)'.format(*pair))
    
    #prompt = 'Finished, press enter to continue' if i+1==len(pairs) else\
    #    'Finished tests for {}/{}. Continue to {}/{}? (y/n): '.format(*pair, *pairs[i+1])
    
    #action = input(prompt)
    
    #if action[0].lower() == 'n':
    #    break
'''        

#cnyusd = daily_test(CNYUSD, plot_equity=True)


#euro_dollar = get_data_yf(tickers='EURUSD=X', interval='1d', period='max', format_for='manual')
#pd.read_csv('Data/Old/EUR_USD.csv').drop(["Vol.", "Change %"], axis=1)[::-1]

#aud_dollar = get_data_yf(tickers='AUDUSD=X', interval='1d', period='max', format_for='manual')
#gbp_dollar = get_data_yf(tickers='GBPUSD=X', interval='1d', period='max', format_for='manual')

#run_test(euro_dollar, 'EUR/USD Low5_RSI2', benchmark=None, plot_equity=True)

#euro_dollar.index = [i for i in range(len(euro_dollar))]
#aud_dollar.index = [i for i in range(len(aud_dollar))]
#gbp_dollar.index = [i for i in range(len(gbp_dollar))]
#snp.index = [i for i in range(len(snp))]

#values1 = create_table(opt_universe)
#values2 = create_table(opt_universe)
#values3 = create_table(opt_universe)
#values4 = create_table(opt_universe)

#values1, best_values1 = optimize_strat(euro_dollar, values1)
#values2, best_values2 = optimize_strat(aud_dollar, values2)
#values3, best_values3 = optimize_strat(gbp_dollar, values3)
#values4, best_values4 = optimize_strat(snp, values4)

'''
for x in best_values1:
    vals = values1.iloc[x]
    low_pd, rsi_pd, rsi_threshold, max_dim = int(vals.low_pd), int(vals.rsi_pd), \
                                            int(vals.rsi_threshold), int(vals.max_dim)

    apply_strat(euro_dollar, 1_000, low_pd, rsi_pd, rsi_threshold, max_dim)
    plot_equity_line(euro_dollar, name='EUR/USD (Best params)')


for x in best_values2:
    vals = values2.iloc[x]
    low_pd, rsi_pd, rsi_threshold, max_dim = int(vals.low_pd), int(vals.rsi_pd), \
                                            int(vals.rsi_threshold), int(vals.max_dim)

    apply_strat(aud_dollar, 1_000, low_pd, rsi_pd, rsi_threshold, max_dim)
    plot_equity_line(aud_dollar, name='AUD/USD (Best params)')


for x in best_values3:
    vals = values3.iloc[x]
    low_pd, rsi_pd, rsi_threshold, max_dim = int(vals.low_pd), int(vals.rsi_pd), \
                                            int(vals.rsi_threshold), int(vals.max_dim)

    apply_strat(gbp_dollar, 1_000, low_pd, rsi_pd, rsi_threshold, max_dim)
    plot_equity_line(gbp_dollar, name='GBP/USD (Best params)')

for x in best_values4:
    vals = values4.iloc[x]
    low_pd, rsi_pd, rsi_threshold, max_dim = int(vals.low_pd), int(vals.rsi_pd), \
                                            int(vals.rsi_threshold), int(vals.max_dim)

    apply_strat(snp, 1_000, low_pd, rsi_pd, rsi_threshold, max_dim)
    plot_equity_line(snp, name='S&P 500 (Best params)')
'''
