#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 02:08:58 2024

@author: aniruddh
"""

from utils import time_utc
import datetime as dt

class Timing():
    def __init__(self, last_candle):
        self.last_candle = last_candle
        if last_candle is None:
            self.last_candle = time_utc() - dt.timedelta(minutes=10)
        self.ready = False
        
    def __repr__(self):
        return str(vars(self))