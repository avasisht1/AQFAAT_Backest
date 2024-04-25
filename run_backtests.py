#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:07:11 2024

@author: aniruddh
"""

from data_puller import get_data_yf
import pandas as pd
import numpy as np
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

train_prop = (7,10)
num_pairs = len(pairs)

folder_path = 'vbt_strats'
if __name__ == '__main__':
    for pair in pairs:
        #default = (5, 2, 50, 5)
        # Get the actual data for the pair
        #pair = pairs[i]
        name = pair[0]+pair[1]+'=X'
        # Change the following to reading from csv after it's all been stored
        df = get_data_yf(tickers=name, interval='1h', period='2y')
        #print('Reading data from Data/vbt/{}.csv'.format(name))
        #df = pd.read_csv('Data/vbt/{}.csv'.format(name))
        
        split_point = len(df.index)*train_prop[0]//train_prop[1]
        df_train = df.iloc[:split_point]
        df_test = df.iloc[split_point:]
        
        sames_train = np.where(df_train['High'] == df_train['Low'])[0]
        df_train = df_train.drop([df_train.index[i] for i in sames_train], axis=0)
        
        sames_test = np.where(df_test['High'] == df_test['Low'])[0]
        df_test = df_test.drop([df_test.index[i] for i in sames_test], axis=0)
        
        # Remove later after all data has been stored
        #df.to_csv('Data/vbt/{}.csv'.format(name))
        #print('Historical data written to Data/vbt/{}.csv'.format(name))
        
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
            
            filename_train = 'metrics/strat{}/hourly/{}_{}_{}.csv'.format(strat_number, name, *train_prop)
            print('Done! Writing training metrics to {}'.format(filename_train))
            metrics_train.to_csv(filename_train)
            print('Train metrics written to {}'.format(filename_train))
            
            test_prop = (train_prop[1]-train_prop[0], train_prop[1])
            filename_test = 'metrics/strat{}/hourly/{}_{}_{}.csv'.format(strat_number, name, *test_prop)
            
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