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
import matplotlib.pyplot as plt

opt_universe = {"low_pd":[3,8,1], "rsi_pd":[2,6,1], "rsi_threshold":[40,71,10], "max_dim":[5,7,1]}

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
    values['Win Rate'] = [0.0 for i in range(n)]
    values['Result'] = [0.0 for i in range(n)]
    for i in range(n):
        combo = list(values[names].iloc[i])
        df, metrics = apply_strat(df, init_capital, *combo, keep_cols=False)
        tdim, n, nrt, nw, result = tuple(metrics.values())
        values.loc[i, 'Result'] = result
        values.loc[i, 'Days in Market'] = tdim
        values.loc[i, 'In-market pct'] = 100 * tdim / n
        values.loc[i, '# Rnd Trips'] = nrt
        values.loc[i, 'Win Rate'] = nw / n
    best_values = np.where(values['Result']==max(values['Result']))
    print("Best Result: {} from the following parameters".format(max(values['Result'])))
    for val in best_values:
        print(values[names].iloc[val])
    return values, list(best_values[0])
        
#values = create_table(opt_universe)
#optimize_strat(aud_dollar, values)

def plot_search_space(df, values, param1, param2, param3_tup, param4_tup):
    # values is already optimized using opt_universe
    
    param1_values = range(*opt_universe[param1])  # Adjust the range and number of points as needed
    param2_values = range(*opt_universe[param2])
    n = len(param1_values)
    m = len(param2_values)
    param3, param3_val = param3_tup
    param4, param4_val = param4_tup

    # Create a meshgrid for the parameters
    param1_mesh, param2_mesh = np.meshgrid(param1_values, param2_values)
    #print('n={}, m={}'.format(n,m))
    #print(param1_mesh)
    #print(param2_mesh)
    
    new = values.where(values[param3]==param3_val).where(values[param4]==param4_val).dropna()
    #print(param1_mesh)
    #print(param2_mesh)
    def result(i, j):
        target = new['Result'].where(new[param1]==param1_mesh[i][j])\
            .where(new[param2]==param2_mesh[i][j]).dropna()
        #print(target)
        return target.iloc[0]
    results = [[result(i,j) for j in range(n)] for i in range(m)]
    best = (param1_mesh[0][0], param2_mesh[0][0], results[0][0])
    for i in range(m):
        for j in range(n):
            if results[i][j]>best[2]:
                best = (param1_mesh[i][j], param2_mesh[i][j], results[i][j])
    print('Best = {}\n'.format(best))
    results = np.array(results)
    
    #print(results)
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surface = ax.plot_surface(param1_mesh, param2_mesh, results, cmap='viridis')

    # Customize the plot
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_zlabel('Final Capital')
    ax.set_title('Final Capital as a Function of {} and {} ({}={}, {}={})'\
                 .format(param1, param2, param3, param3_val, param4, param4_val))

    # Add a colorbar
    fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10)

    # Show the plot
    plt.show()
    return best[0], best[1]
    
#values = create_table(opt_universe)
#optimize_strat(euro_dollar, values)
#plot_search_space(euro_dollar, values, 'rsi_pd', 'low_pd', ('rsi_threshold',50), ('max_dim',5))
    