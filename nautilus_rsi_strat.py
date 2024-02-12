from nautilus_trader.core.message import Event
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.indicators.donchian_channel import DonchianChannels
from nautilus_trader.indicators.average.moving_average import MovingAverageType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.events import PositionOpened
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy, StrategyConfig

'''
This file started as a copy of the code from the same source as
nautilus_learn.py and I've been trying to change it so that it fits my needs.
Eventually the hope is that this code will replace the code in backtest.py
in which I have implemented the indicators and the strategy by hand using just
pandas.
'''


class Low5_RSI2_Config(StrategyConfig):
    instrument_id: InstrumentId
    period: int = 2
    ma_type: int = MovingAverageType.SIMPLE
    low_period: int = 5
    days_in_market: int = 0
    trade_size: int = 100_000
    #entry_threshold: float = 0.00010


class Low5_RSI2_Strategy(Strategy):
    def __init__(self, config: Low5_RSI2_Config) -> None:
        super().__init__(config=config)
        # Our "trading signal"
        self.rsi = RelativeStrengthIndex(
            period=config.period,
            ma_type=config.ma_type
        )
        # We copy some config values onto the class to make them easier to reference later on
        self.low5 = DonchianChannels(5).lower
        #self.entry_threshold = config.entry_threshold
        self.instrument_id = config.instrument_id
        self.trade_size = Quantity.from_int(config.trade_size)

        # Convenience
        self.position: Position | None = None

    def on_start(self):
        self.subscribe_quote_ticks(instrument_id=self.instrument_id)

    def on_stop(self):
        self.close_all_positions(self.instrument_id)
        self.unsubscribe_quote_ticks(instrument_id=self.instrument_id)

    def on_quote_tick(self, tick: QuoteTick):
        # You can register indicators to receive quote tick updates automatically,
        # here we manually update the indicator to demonstrate the flexibility available.
        self.rsi.handle_quote_tick(tick)
        self.low5.handle_quote_tick(tick)

        if not self.rsi.initialized or not self.low5.initialized:
            return  # Wait for indicator to warm up
        
        # self._log.info(f"{self.macd.value=}:%5d")
        self.check_for_entry()
        self.check_for_exit()

    def on_event(self, event):
        if isinstance(event, PositionOpened):
            self.position = self.cache.position(event.position_id)

    def check_for_entry(self):
        # If MACD line is above our entry threshold, we should be LONG
        if self.low5 < 25:# FIGURE OUT HOW TO GET THE 5-DAY LOW
            if self.position and self.position.side == PositionSide.LONG:
                self.days_in_market += 1
                return  # Already LONG

            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.trade_size,
            )
            self.submit_order(order)
        # If MACD line is below our entry threshold, we should be SHORT
        '''
        elif self.macd.value < -self.entry_threshold:
            if self.position and self.position.side == PositionSide.SHORT:
                return  # Already SHORT

            order = self.order_factory.market(
                instrument_id=self.instrument_id,
                order_side=OrderSide.SELL,
                quantity=self.trade_size,
            )
            self.submit_order(order)
        '''

    def check_for_exit(self):
        # If MACD line is above zero then exit if we are SHORT
        if self.rsi.value >= 50.0 or self.days_in_market == 5:
            if self.position and self.position.side == PositionSide.LONG:
                self.close_position(self.position)
        # If MACD line is below zero then exit if we are LONG
        else:
            if self.position and self.position.side == PositionSide.LONG:
                self.close_position(self.position)

    def on_dispose(self):
        pass  # Do nothing else
