#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:01:33 2024

@author: aniruddh
"""

import pandas as pd
import numpy as np
import itertools
from backtest import apply_strat

opt_universe = {"low_pd":[3,7,1], "rsi_pd":[2,6,1], "rsi_threshold":[40,71,10], "max_dim":[5,6,1]}

def create_table(univ):
    """
    Creates a table of all the possible hyperparameter values.
    
    Given a dictionary with the hyperparameter names and an indexable object 
    containing 3 numbers, generates a table where each row is one combination of
    the hyperparameters. The numbers are interpreted int he same way as in a 
    slice or range object with the interval [start, end) being the first two, 
    and the third being the increment.

    Parameters:
        - univ (dict): The dictionary

    Returns:
        pd.DataFrame: Dataframe with one row per combination of hyperparameter values
    """
    
    param_ranges = [range(*univ[x]) for x in univ]
    combos = []
    for combo in itertools.product(*param_ranges):
        combos.append(combo)
    values = pd.DataFrame(combos)
    values.columns = univ.keys()
    return values

def optimize_strat(df, values, init_capital=1000):
    """
    Finds the best set of hyperparameter values as well as statistics about the
    various combinations
    
    Given a dataframe with historical data, a table of values, and optionally an
    amount of initial capital, implements the strategy from apply_strat with
    all the combinations of the hyperparameters, updates the values dataframe
    with relevant statistics, and prints the best combination to the console.

    Parameters:
        - df (pd.DataFrame): The dataframe of historical data
        - values (pd.DataFrame): The values table
        - [init_capital] (float): The starting capital for the strategy

    Returns:
        pd.DataFrame: Dataframe with one row per combination of hyperparameter,
        now with relevant statistics
    """
    
    n = values.shape[0]
    names = values.columns
    values['Days in Market'] = [0 for i in range(n)]
    values['In-market pct'] = [0.0 for i in range(n)]
    values['# Rnd Trips'] = [0 for i in range(n)]
    values['Result'] = [0.0 for i in range(n)]
    for i in range(n):
        combo = list(values[names].iloc[i])
        df, metrics = apply_strat(df, init_capital, *combo, keep_cols=False)
        tdim, n, nrt, nw, result = tuple(metrics.values())
        values.loc[i, 'Result'] = result
        values.loc[i, 'Days in Market'] = tdim
        values.loc[i, 'In-market pct'] = 100 * tdim / n
        values.loc[i, '# Rnd Trips'] = nrt
    best_values = np.where(values['Result']==max(values['Result']))
    print("Best Result: {} from the following parameters".format(max(values['Result'])))
    for val in best_values:
        print(values[names].iloc[val])
    return values, list(best_values[0])
        
#values = create_table(opt_universe)
#optimize_strat(aud_dollar, values)

