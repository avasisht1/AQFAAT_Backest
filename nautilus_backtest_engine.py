#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 11:14:41 2024

@author: aniruddh
"""

'''
This file started as a copy of the code from the nautilus_trader github repo
(filepath = nautlius_trader/examples/backtest/fx_ema_cross_audusd_bars_from_ticks.py)
and I've been trying to change it so that it fits my needs.
'''

import time
from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.backtest.engine import BacktestEngineConfig
from nautilus_trader.backtest.modules import FXRolloverInterestConfig
from nautilus_trader.backtest.modules import FXRolloverInterestModule
#from nautilus_trader.examples.strategies.ema_cross import EMACross
#from nautilus_trader.examples.strategies.ema_cross import EMACrossConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType
from nautilus_trader.model.enums import OmsType
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestDataProvider
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_rsi_strat import LowX_RSIY_Strategy
from nautilus_rsi_strat import LowX_RSIY_Config
from nautilus_trader.indicators.average.moving_average import MovingAverageType
from data_puller import get_data_av, get_data_yf
import matplotlib.pyplot as plt
from backtest import plot_equity_line


if __name__ == "__main__":
    # Configure backtest engine
    config = BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
    )

    # Build the backtest engine
    engine = BacktestEngine(config=config)

    # Optional plug in module to simulate rollover interest,
    # the data is coming from packaged test data.
    provider = TestDataProvider()
    interest_rate_data = provider.read_csv("short-term-interest.csv")
    config = FXRolloverInterestConfig(interest_rate_data)
    fx_rollover_interest = FXRolloverInterestModule(config=config)

    # Add a trading venue (multiple venues possible)
    SIM = Venue("SIM")
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.NETTING,  # Venue will generate position IDs
        account_type=AccountType.MARGIN,
        base_currency=USD,  # Standard single-currency account
        starting_balances=[Money(1_000_000, USD)],  # Single-currency or multi-currency accounts
        modules=[fx_rollover_interest],
    )

    # Add instruments
    EURUSD_SIM = TestInstrumentProvider.default_fx_ccy("EUR/USD", SIM)
    engine.add_instrument(EURUSD_SIM)

    # Add data
    wrangler = BarDataWrangler(instrument=EURUSD_SIM, bar_type=BarType.from_str("EUR/USD.SIM-1-DAY-MID-EXTERNAL"))
    euro_dollar_full_1d = get_data_av(('EUR','USD'), 'full', 'nautilus', 'FX_DAILY', '1d')
    #print(euro_dollar_full_1d)
    bars = wrangler.process(euro_dollar_full_1d)
    engine.add_data(bars)

    # Configure your strategy
    config = LowX_RSIY_Config(
        rsi_period = 2,
        bar_type=BarType.from_str("EUR/USD.SIM-1-DAY-MID-EXTERNAL"),
        ma_type = MovingAverageType.EXPONENTIAL,
        low_period = 5,
        max_days_in_market = 5,
        trade_size = Decimal(1_000_000),
        instrument_id = EURUSD_SIM.id
    )
    # Instantiate and add your strategy
    strategy = LowX_RSIY_Strategy(config=config)
    engine.add_strategy(strategy=strategy)

    time.sleep(0.1)
    input("Press Enter to continue...")

    # Run the engine (from start to end of data)
    engine.run()
    
    print("\n\n**EVERYTHING ABOVE THIS IS ENGINE OUTPUT**\n\n")

    # Optionally view reports
    '''with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        None,
        "display.width",
        300,
    ):'''
    
    account_report = engine.trader.generate_account_report(SIM)
    fills_report = engine.trader.generate_fills_report()
    order_fill_report = engine.trader.generate_order_fills_report()
    orders_report = engine.trader.generate_orders_report()
    positions_report = engine.trader.generate_positions_report()

    # For repeated backtest runs make sure to reset the engine
    engine.reset()

    # Good practice to dispose of the object when done
    plt.plot(account_report['total'].astype(float))
    plt.show()
    engine.dispose()
    
    