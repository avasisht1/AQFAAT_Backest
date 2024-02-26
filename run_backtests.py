#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:07:11 2024

@author: aniruddh
"""

from data_puller import get_data_av, get_data_yf, format_manual, format_nautilus
from opt_universe import optimize_strat, create_table
from backtest import run_test, hourly_to_daily, plot_equity_line
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

def hourly_test(pair, plot_equity=False):
    df = get_data_av(pair, 'compact', format_for='manual')
    run_test(df, 1_000, plot_equity=plot_equity)
    
    
def daily_test(pair, benchmark, plot_equity=False):
    df = get_data_av(pair, 'full', format_for='manual')
    df_daily = hourly_to_daily(df)
    name = '{}/{}'.format(*pair)
    run_test(df_daily, 'RSI2, Low5 for {}'.format(name), benchmark, 1_000, plot_equity=plot_equity)
    return df_daily

#cnyusd = daily_test(CNYUSD, plot_equity=True)

opt_universe = {"low_pd":[3,7,1], "rsi_pd":[2,6,1], "rsi_threshold":[40,71,10], "max_dim":[5,6,1]}

euro_dollar = get_data_yf(tickers='EURUSD=X', interval='1d', period='max', format_for='manual')
#pd.read_csv('Data/Old/EUR_USD.csv').drop(["Vol.", "Change %"], axis=1)[::-1]

aud_dollar = get_data_yf(tickers='AUDUSD=X', interval='1d', period='max', format_for='manual')
gbp_dollar = get_data_yf(tickers='GBPUSD=X', interval='1d', period='max', format_for='manual')
snp = pd.read_csv('Data/YahooFinance/SPX.csv').drop(['Adj Close', 'Volume'], axis=1)
snp = format_manual(snp)
benchmark=(snp, 'Buy&Hold S&P 500', True)

run_test(euro_dollar, 'EUR/USD Low5_RSI2', benchmark=None,
         plot_equity=True)

#euro_dollar.index = [i for i in range(len(euro_dollar))]
#aud_dollar.index = [i for i in range(len(aud_dollar))]
#gbp_dollar.index = [i for i in range(len(gbp_dollar))]
#snp.index = [i for i in range(len(snp))]

values1 = create_table(opt_universe)
values2 = create_table(opt_universe)
values3 = create_table(opt_universe)
values4 = create_table(opt_universe)

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
