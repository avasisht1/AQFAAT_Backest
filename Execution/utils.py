#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:48:56 2024

@author: aniruddh
"""

import os
import datetime as dt
from dateutil.parser import *

path = os.getcwd()
parent = os.path.abspath(os.path.join(path, os.pardir))

def get_his_data_filename(pair, granularity):
    filename = '{}/Data/Oanda/{}_{}.pkl'.format(parent, pair, granularity)
    if os.path.isfile(filename):
        return filename
    else:
        print('File not found')
        return

def get_instruments_data_filename():
    return 'instruments.pkl'


def time_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def get_utc_dt_from_string(date_str):
    d = parse(date_str)
    return d.replace(tzinfo=dt.timezone.utc)

if __name__ == '__main__':
    print(get_his_data_filename('EUR_USD', 'M1'))
    print(get_instruments_data_filename())