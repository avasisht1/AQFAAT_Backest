#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:13:38 2024

@author: aniruddh
"""

API_KEY = '194a9b1f38a1e004c55a95089e447553-36c0ce325cce3cd043cbd6bd748b93ab'
ACCOUNT_ID = '101-001-28995934-001'
OANDA_URL = 'https://api-fxpractice.oanda.com/v3'

SECURE_HEADER = {
    'Authorization': 'Bearer {}'.format(API_KEY),
    'Content-Type': 'application/json'
}
# Oanda API Key = 194a9b1f38a1e004c55a95089e447553-36c0ce325cce3cd043cbd6bd748b93ab
# Oanda Acct # = 101-001-28995934-001

BUY = 1
SELL = -1
NONE = 0