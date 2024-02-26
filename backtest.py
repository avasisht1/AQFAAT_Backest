import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from data_puller import get_data_av, get_data_yf

'''
euro_dollar = pd.read_csv('Old/EUR_USD.csv').drop(["Vol.", "Change %"], axis=1)[::-1]
aud_dollar = pd.read_csv('Old/AUD_USD.csv').drop(["Vol.", "Change %"], axis=1)[::-1]
gbp_dollar = pd.read_csv('Old/GBP_USD.csv').drop(["Vol.", "Change %"], axis=1)[::-1]
snp = pd.read_csv('SPX.csv').drop(['Adj Close', 'Volume'], axis=1)

euro_dollar.index = [i for i in range(len(euro_dollar))]
aud_dollar.index = [i for i in range(len(aud_dollar))]
gbp_dollar.index = [i for i in range(len(gbp_dollar))]
'''

def validate_data(dataframe, error_code="Date"):
    """
    Makes sure that a dataframe of historical OHLC data are valid.

    Given a dataframe and a column to be used to report inconsistencies,
    prints the data in that column if any are found. Inconsistencies are when
    the high is less than the low.

    Parameters:
        - dataframe (int): The first number.
        - b (int): The second number.

    Returns:
        int: The sum of a and b.
    """
    
    for i in range(len(dataframe["Open"])):
        if dataframe["High"][i] < dataframe["Low"][i]:
            print("Bad data at {}".format(dataframe[error_code]))


def compare_data_column(df1, df2, column=["Open", "Open"], error_code="Date"):
    """
    Makes sure that two dataframes of historical data have the same data.

    Given two dataframes and a column to be used to report discrepancies, 
    prints the data in that column if any are found. This function is not used
    in the following code but may be useful later when multiple data sources
    are cross-referenced.

    Parameters:
        - df1 (pd.DataFrame): The first dataframe.
        - df2 (pd.DataFrame): The second dataframe.
        - column (List-like indexable object): The columns of the two dataframes
        being compared
        - error_code (str): The name of the column used to report discrepancies

    Returns:
        None, but prints to the console
    """
    
    n1 = len(df1[column[0]])
    n2 = len(df2[column[1]])
    n = min(n1, n2)
    for i in range(0, n, -1):
        idx1 = n1 - i
        idx2 = n2 - i
        if df1[column[0]][idx1] != df2[column[1]][idx2]:
            print("Conflicting Data at {} / {}".format(df1[error_code], df2[error_code]))
    

def hourly_to_daily(hourly_ohlc):
    daily = pd.DataFrame(
    {'Open': hourly_ohlc['Open'].resample('24h').first(),
    'High': hourly_ohlc['High'].resample('24h').max(),
    'Low': hourly_ohlc['Low'].resample('24h').min(),
    'Close': hourly_ohlc['Close'].resample('24h').last()}
    ).dropna()
    
    return daily


def calculate_nday_low(dataframe, column='Low', n=5):
    """
    Calculates the low over a specified period using a given column of the input
    dataframe.

    Given a dataframe, a column to compare, and a period on which to compare, 
    appends a new column to the dataframe with the low of the specified column
    over the window.

    Parameters:
        - dataframe (pd.DataFrame): The dataframe
        - column (str): The column of which to calculate lows
        - n (int): The period on which the low should be calculated

    Returns:
        pd.DataFrame: The input dataframe with another column containing the
        n-day lows of the specified column.
    """
    
    closes = dataframe[column]
    lows = closes.rolling(window=n, min_periods=1).min()
    dataframe["Low_" + str(n)] = lows
    return dataframe


def calculate_rsi(dataframe, column='Close', window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given DataFrame.

    Parameters:
    - dataframe: pandas DataFrame containing financial data.
    - column: Name of the column containing the closing prices. Default is 'Close'.
    - window: Lookback period for calculating RSI. Default is 14.

    Returns:
    - dataframe: Original DataFrame with an additional 'RSI' column.
    """

    # Calculate price changes
    delta = dataframe[column].pct_change()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses over the specified window
    avg_gains = list(gains.rolling(2).mean())
    avg_losses = list(losses.rolling(2).mean())
    avg_gains[0] = gains.iloc[0]
    avg_losses[0] = losses.iloc[0]

    # Calculate relative strength (RS) and RSI
    rsi = [0 for _ in range(len(dataframe[column]))]
    
    for i in range(len(dataframe[column])):
        if i == 0:
            rsi[i] = 100
        elif i < window:
            if avg_losses[i-1] != 0:
                rs0 = avg_gains[i-1] / avg_losses[i-1]
                rsi[i] = 100 - (100 / (1 + rs0))
            else:
                rsi[i] = 100
        else: # i > window
            avg_gains[i] = (avg_gains[i-1]*(window-1) + gains.iloc[i])/window
            avg_losses[i] = (avg_losses[i-1]*(window-1) + losses.iloc[i])/window
            if avg_losses[i] != 0:
                rs = avg_gains[i]/avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
                
    #dataframe["Avg_gains"] = avg_gains
    #dataframe["Avg_losses"] = avg_losses
    #dataframe["Gains"] = gains
    #dataframe["Losses"] = losses

    # Add 'RSI' column to the original DataFrame
    dataframe['RSI_'+str(window)] = rsi

    return dataframe


def plot_candlestick_rsi(dataframe):
    """
    Plot OHLC candlesticks and daily RSI using matplotlib.

    Parameters:
    - dataframe: pandas DataFrame containing daily market data with 'Date', 'Open', 'High', 'Low', 'Close', 'Adj. Close', 'Volume', and 'RSI' columns.
    """
    
    ohlc = dataframe[['Date', 'Open', 'High', 'Low', 'Close']]
    rsi = dataframe['RSI_2']

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 8))
    
    # Plot candlesticks
    ohlc['Date'] = pd.to_datetime(ohlc['Date'])
    ohlc['Date'] = ohlc['Date'].apply(mdates.date2num)
    candlestick_ohlc(ax1, ohlc.values, width=0.6, colorup='g', colordown='r')
    ax1.set_ylabel('OHLC')

    # Plot RSI
    ax2.plot(ohlc['Date'], rsi, color='purple', label='2-day RSI')
    ax2.axhline(70, color='r', linestyle='--', label='Overbought (70)')
    ax2.axhline(30, color='g', linestyle='--', label='Oversold (30)')
    ax2.set_ylabel('RSI')
    ax2.legend()

    # Format x-axis as dates
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.show()
    

def plot_equity_line(df, name='', cols=['Equity']):
    """
    Plots the equity line of a given dataframe given a column to plot.
    
    Given a dataframe, a strategy name, and a column (default: "Equity"), this
    function plots the data in the given column as a function of the row it
    came from.

    Parameters:
        - dataframe (pd.DataFrame): The dataframe
        - col (str): The column to plot

    Returns:
        None: But plots the data
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(df[cols], label='Equity Curve')
    title = "Equity Curve for {}".format(name) if name!="" else "Equity Curve"
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()


def apply_strat(dataframe, init_capital, low_pd=5, rsi_pd=2, rsi_threshold=50, max_dim=5,
                keep_cols=True):
    """
    Implements the third strategy in this article 
    https://www.quantifiedstrategies.com/3-free-mean-reversion-trading-strategies/
    But in more generality.
    
    Given a dataframe with daily OHLC market data, an initial amount of capital,
    a low period, an rsi period, an rsi threshold, and a maximum number of days 
    to stay in the market, creates two new columns for the action taken and the
    equity of the strategy based on a strategy with the rules described in the 
    article. This adds the columns for the n-day low and the n-day RSI as described
    in calculate_rsi and calculate_nday_low.

    Parameters:
        - dataframe (pd.DataFrame): The dataframe
        - init_capital (int or float): The starting capital
        - low_pd (int): The number of days used to calculate the low for the
        entry point
        - rsi_pd (int): The number of days used to calculate the rsi for the
        exit point
        - max_dim (int): The maximum number of days to stay in the market as a
        stop-loss condition
        - keep_cols (bool): Option to keep new columns added. If False, will
        drop all new columns added to the dataframe

    Returns:
        pd.DataFrame: Dataframe with the new columns
        int: The number of days we had a position in the market under this strategy
        int: The total number of days of historical data in the dataframe
        int: The total number of times we entered and exited the market
        float: The final amount of capital after executing the strategy over the
        period of the historical data.
    """
    
    calculate_rsi(dataframe, column="Close", window=rsi_pd)
    calculate_nday_low(dataframe, column="Low", n=low_pd)
    
    n = len(dataframe["Close"])
    total_days_in_market = 0
    num_round_trips = 0
    curr_capital = init_capital
    in_market = False
    days_in_market = 0
    equity_line = [init_capital for _ in range(n)]
    action = ["None" for _ in range(n)]
    for i in range(max(low_pd, rsi_pd), n):
        curr_close = dataframe["Close"].iloc[i]
        assert(type(curr_close) == np.float64)
        curr_low = dataframe["Low_" + str(low_pd)].iloc[i - 1]
        curr_rsi = dataframe["RSI_" + str(rsi_pd)].iloc[i]
        if not in_market:
            if curr_close < curr_low:
                in_market = True
                action[i] = "Enter"
        else: # in_market
            assert(type(dataframe['Close'].iloc[i-1]) == np.float64)
            curr_capital *= curr_close / dataframe["Close"].iloc[i - 1]
            if curr_rsi > rsi_threshold or days_in_market == max_dim:
                in_market = False
                action[i] = "Exit"
                total_days_in_market += days_in_market
                days_in_market = 0
                num_round_trips += 1
            else: # curr_rsi <= rsi_threshold and days_in_market < max_dim
                days_in_market += 1
        equity_line[i] = curr_capital
    dataframe["Equity"] = equity_line
    dataframe["Action"] = action
    num_wins = 0
    for i in range(1, len(dataframe.index), 2):
        if dataframe['Action'].iloc[i] > dataframe['Action'].iloc[i-1]:
            num_wins += 1
    if not keep_cols:
        dataframe = dataframe.drop(['Action', 'RSI_'+str(rsi_pd), 'Low_'+str(low_pd)], axis=1)
    
    stats = {'Total Days in Market':total_days_in_market, 'Total days':n,
             'Number of Round Trips':num_round_trips, 'Wins':num_wins,
             'Final Capital':equity_line[-1]}
    return dataframe, stats


def calculate_sharpe_ratio(df, benchmark_equity, k=10):
    equity = df['Equity']
    n = len(equity)
    returns = []
    benchmark_returns = []
    for i in range(1,k+1):
        idx = i * (n//k)
        prev = (i-1) * (n//k)
        curr_equity = df['Equity'].iloc[idx]
        prev_equity = df['Equity'].iloc[prev]
        curr_benchmark = benchmark_equity['BH Equity'].iloc[idx]
        prev_benchmark = benchmark_equity['BH Equity'].iloc[prev]
        
        strat_rtn = (curr_equity - prev_equity)/prev_equity
        benchmark_rtn = (curr_benchmark - prev_benchmark)/prev_benchmark
        
        returns.append(strat_rtn)
        benchmark_returns.append(benchmark_rtn)
    
    R_p = sum(returns)/len(returns)
    R_f = sum(benchmark_returns)/len(benchmark_returns)
    sd = (pd.Series(returns) - pd.Series(benchmark_returns)).var()
    vol = pd.Series(returns).std() * np.sqrt(k)
    return (R_p - R_f)/sd, vol


def calculate_drawdown(equity_series):
    # Calculate the cumulative returns
    cumulative_returns = (1 + equity_series.pct_change()).cumprod()

    # Calculate the cumulative maximum value
    cumulative_max = cumulative_returns.cummax()

    # Calculate the drawdown
    drawdown = (cumulative_max - cumulative_returns)

    # Find the maximum drawdown and its start and end dates
    max_drawdown = drawdown.max()
    max_drawdown_start = drawdown.idxmax()
    max_drawdown_end = cumulative_returns.idxmax()

    return max_drawdown, max_drawdown_start, max_drawdown_end


def buy_and_hold(dataframe, init_capital):
    """
    Implements a buy-and-hold strategy for the instrument whose data is in the
    given dataframe with a given initial capital.
    
    Given a dataframe with daily OHLC market data and an initial amount of capital,
    creates a new dataframe with two columns for date and the equity of a buy-
    and-hold strategy in the market whose data is in the input dataframe.

    Parameters:
        - dataframe (pd.DataFrame): The dataframe
        - init_capital (int or float): The starting capital

    Returns:
        pd.DataFrame: Dataframe with the two columns for date and equity
        float: Final capital after executing the buy-and-hold strategy over the
        period of the historical data.
    """
    
    n = len(dataframe["Close"])
    curr_capital = init_capital
    equity_line = [init_capital for i in range(n)]
    for i in range(1, n):
        curr_close = dataframe["Close"].iloc[i]
        prev_close = dataframe["Close"].iloc[i - 1]
        curr_capital *= curr_close / prev_close
        equity_line[i] = curr_capital
    dataframe.insert(len(dataframe.columns), 'BH Equity', equity_line, True)
    return dataframe['BH Equity'], equity_line[-1]
    

def run_test(df, name, init_capital=1000, benchmark=None, plot_ohlc_rsi=False, plot_equity=False):
    """
    Runs a backtest on the strategy implemented in apply_strat.
    
    Given a dataframe with daily OHLC market data and an initial amount of capital,
    runs the strategy from apply_strat and optionally plots the equity, rsi, and
    calculates and plots the buy-and-hold equity time series.

    Parameters:
        - df (pd.DataFrame): The dataframe
        - init_capital (int or float): The starting capital
        - plot_ohlc_rsi (bool): Flag indicating whether or not to plot the data
        in df as well as the rsi
        - plot_equity (bool): Flag indicating whether or not to plot the equity
        line of the strategy
        - plot_buy_and_hold (bool): Flag indicating whether or not to calculate
        the equity of the buy-and-hold strategy and plot it

    Returns:
        pd.DataFrame: Dataframe with the additional columns acquired by apply_strat
    """
    
    out, metrics = apply_strat(df, init_capital, keep_cols=False)
    if plot_ohlc_rsi:
        plot_candlestick_rsi(df)
        
    nrt, num_wins, tdim, n, final = tuple(metrics.values())
    peak = max(out['Equity'])
    metrics['Equity Peak'] = peak
    final_capital = metrics['Final Capital']
    num_years = (out.index[-1] - out.index[0]).days/252
    metrics['Annualized Return'] = ((final_capital/init_capital) - 1)/num_years
    #metrics['Max Drawdown'] = calculate_drawdown(out['Equity'])
    
    if not (benchmark is None) and not (benchmark[0] is None):
        benchmark, benchmark_name, plot_benchmark = benchmark
        benchmark = benchmark.loc[max(df.index[0], benchmark.index[0]):min(df.index[-1], benchmark.index[-1])]
        outbh, finalbh = buy_and_hold(benchmark, init_capital)
        df.insert(len(df.columns), 'BH Equity', outbh, True)
        print("Buy and Hold Final Capital: {}".format(finalbh))
        metrics['Sharpe Ratio'], metrics['Volatility'] = calculate_sharpe_ratio(df, df)
    else:
        benchmark_name = ''
        plot_benchmark = False
    
    if plot_equity:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Equity'], label='Equity Curve')
        title=''
        if name=='':
            title = 'Equity Curve'
        else:
            if benchmark_name=='':
                title = 'Equity Curve for {}'.format(name)
            else:
                title = 'Equity Curve for {} vs {}'.format(name, benchmark_name)
        plt.title(title)
        plt.xlabel('Timestamp')
        plt.ylabel('Equity')
        if plot_benchmark:
            plt.plot(outbh, label=benchmark_name)
        plt.legend()
        
        #plt.axvline(metrics['Max Drawdown'][1])
        #plt.axvline(metrics['Max Drawdown'][2])
        
        plt.grid(True)
        plt.show()
        
        print('Metrics for Strategy {}:\n{}'.format(name, metrics))
    return df, metrics

