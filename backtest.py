import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.average.moving_average import MovingAverageType


euro_dollar = pd.read_csv('EUR_USD Historical Data.csv').drop(["Vol.", "Change %"],axis=1)[::-1]
aud_dollar = pd.read_csv('AUD_USD Historical Data.csv').drop(["Vol.", "Change %"],axis=1)[::-1]
gbp_dollar = pd.read_csv('GBP_USD Historical Data.csv').drop(["Vol.", "Change %"],axis=1)[::-1]
snp = pd.read_csv('SPX.csv').drop(['Adj Close', 'Volume'],axis=1)

euro_dollar.index = [i for i in range(len(euro_dollar))]
aud_dollar.index = [i for i in range(len(aud_dollar))]
gbp_dollar.index = [i for i in range(len(gbp_dollar))]

#euro_dollar2 = pd.read_csv("EURUSD2.csv")
#aud_dollar2 = pd.read_csv("AUDUSD2.csv")
#gbp_dollar2 = pd.read_csv("GBPUSD2.csv")

#euro_dollar["Pct Change"] = (euro_dollar["Close"]/euro_dollar["Open"])-1
#aud_dollar["Pct Change"] = (aud_dollar["Close"]/aud_dollar["Open"])-1
#gbp_dollar["Pct Change"] = (gbp_dollar["Close"]/gbp_dollar["Open"])-1

rsi2 = RelativeStrengthIndex(2, ma_type = MovingAverageType.SIMPLE)


def validate_data(dataframe, error_code="Date"):
    for i in range(len(dataframe["Open"])):
        if dataframe["High"][i] < dataframe["Low"][i]:
            print("Bad data at {}".format(dataframe[error_code]))

            
def compare_data_column(df1, df2, column=["Open", "Open"], error_code="Date"):
    n1 = len(df1[column[0]])
    n2 = len(df2[column[1]])
    n = min(n1, n2)
    for i in range(0, n, -1):
        idx1 = n1 - i
        idx2 = n2 - i
        if df1[column[0]][idx1] != df2[column[1]][idx2]:
            print("Conflicting Data at {} / {}".format(df1[error_code],
                                                       df2[error_code]))


def calculate_nday_low(dataframe, column='Low', n=5):
    closes = dataframe[column]
    lows = closes.rolling(n, 1).min()
    dataframe["Low_"+str(n)] = lows
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
    delta = dataframe[column].diff(1)

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses over the specified window
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()

    # Calculate relative strength (RS) and RSI
    rsi = [0 for _ in range(len(dataframe[column]))]
    try:
        rs0 = avg_gains[window] / avg_losses[window]
        rsi0 = 100 - (100 / (1 + rs0))
    except:
        rsi0 = 100
    
    for i in range(len(dataframe[column])):
        if i < window:
            rsi[i] = np.nan
        elif i == window:
            rsi[i] = rsi0
        else: # i > window
            avg_gains[i] = (avg_gains[i-1]*(window-1) + gains[i])/window
            avg_losses[i] = (avg_losses[i-1]*(window-1) + losses[i])/window
            try:
                rs = avg_gains[i]/avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))
            except:
                rsi[i] = 100
                
    dataframe["Avg_gains"] = avg_gains
    dataframe["Avg_losses"] = avg_losses
    dataframe["Gains"] = gains
    dataframe["Losses"] = losses

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
    

def plot_equity_line(df, name='', col='Equity'):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df[col], label='Equity Curve', color='blue')
    title = "Equity Curve for {}".format(name) if name!="" else "Equity Curve"
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.legend()
    plt.grid(True)
    plt.show()


def apply_strat(dataframe, init_capital, low_pd, rsi_pd, max_dim):
    n = len(dataframe["Close"])
    total_days_in_market = 0
    num_round_trips = 0
    curr_capital = init_capital
    in_market = False
    days_in_market = 0
    equity_line = [init_capital for _ in range(n)]
    action = ["None" for _ in range(n)]
    for i in range(5, n):
        curr_close = dataframe["Close"][i]
        curr_low = dataframe["Low_5"][i-1]
        curr_rsi = dataframe["RSI_2"][i]
        if not in_market:
            if curr_close < curr_low:
                in_market = True
                action[i] = "Enter"
        else: # in_market
            curr_capital *= curr_close/dataframe["Close"][i-1]
            if curr_rsi > 50 or days_in_market == 5:
                in_market = False
                action[i] = "Exit"
                total_days_in_market += days_in_market
                days_in_market = 0
                num_round_trips += 1
            else: # curr_rsi <= 50 and days_in_market < 5
                days_in_market += 1
        equity_line[i] = curr_capital
    dataframe["Equity"] = equity_line
    dataframe["Action"] = action
    return dataframe, total_days_in_market, n, num_round_trips, equity_line[-1]


def buy_and_hold(dataframe, init_capital):
    n = len(dataframe["Close"])
    curr_capital = init_capital
    equity_line = [init_capital for i in range(n)]
    for i in range(1,n):
        curr_close = dataframe["Close"][i]
        prev_close = dataframe["Close"][i-1]
        curr_capital *= curr_close/prev_close
        equity_line[i] = curr_capital
    return pd.DataFrame({"Date": dataframe["Date"], "BH Equity": equity_line}), equity_line[-1]
    
'''
validate_data(euro_dollar)
validate_data(aud_dollar)
validate_data(gbp_dollar)

compare_data_column(euro_dollar, euro_dollar2)
compare_data_column(aud_dollar, aud_dollar2)
compare_data_column(gbp_dollar, gbp_dollar2)

compare_data_column(euro_dollar, euro_dollar2, column=["Close", "Close/Last"])
compare_data_column(aud_dollar, aud_dollar2, column=["Close", "Close/Last"])
compare_data_column(gbp_dollar, gbp_dollar2, column=["Close", "Close/Last"])
'''

def run_test(df, name, plot_ohlc_rsi=False, plot_equity=False, plot_buy_hold=False):
    calculate_rsi(df, column="Close", window=2)
    calculate_nday_low(df)
    
    if plot_ohlc_rsi:
        plot_candlestick_rsi(df)
    
    out, tdim, n, nrt, final = apply_strat(df, 1000)
    
    print("Strategy Name: {}\n\tTime in Market: {}/{} = {}%\n\tFinal Capital: {}"\
          .format(name, tdim, n, 100*tdim/n, final))
    
    if plot_buy_hold:
        outbh, finalbh = buy_and_hold(df, 1000)
        print("Buy and Hold Final Capital: {}".format(finalbh))
        plot_equity_line(outbh, "Buy and Hold", 'BH Equity')
    
    if plot_equity:
        plot_equity_line(df, name=name)
    
    return df

run_test(snp, "S&P 500 RSI2, Low5")


