#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:07:11 2024

@author: aniruddh
"""

from data_puller import get_data_av, get_data_yf, format_manual, format_nautilus
from opt_universe import optimize_strat, create_table, plot_search_space
from backtest import run_test, hourly_to_daily, plot_equity_line, apply_strat
import pandas as pd


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

pairs = [(EUR, USD), (AUD, USD), (GBP, USD), (USD, CAD), (USD, JPY), (USD, CNY), (USD, CHF), (USD, HKD), (EUR, GBP), (USD, KRW)]

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
training_proportion = 2/3


for i in range(2,3):
    #default = (5,T 2, 50, 5)
    # Get the actual data for the pair
    pair = pairs[i]
    name = pair[0]+pair[1]+'=X'
    df = get_data_yf(tickers=name, interval='1d', period='max', format_for='manual')
    split_point = len(df.index)*2//3
    df_train = df.iloc[:split_point]
    df_test = df.iloc[split_point:]
    # Run the tests and plot equity lines with and without the benchmark comparison
    
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
