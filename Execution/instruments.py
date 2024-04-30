#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:20:19 2024

@author: aniruddh
"""

import pandas as pd
import utils

class Instrument():
    def __init__(self, ob):
        self.name = ob['name']
        self.ins_type = ob['type']
        self.displayName = ob['displayName']
        self.pipLocation = 10 ** ob['pipLocation']
        self.marginRate = ob['marginRate']
        
        
    def __repr__(self):
        return str(vars(self))
     
    @classmethod
    def get_instruments_df(cls):
        return pd.read_pickle (utils.get_instruments_data_filename())
    
    @classmethod
    def get_instruments_list(cls):
        df = cls.get_instruments_df()
        return [Instrument(x) for x in df.to_dict(orient='records')]
    
    @classmethod
    def get_instruments_dict(cls):
        i_list = cls.get_instruments_list()
        i_keys = [x.name for x in i_list]
        return {k:v for (k,v) in zip(i_keys, i_list)}
    
    @classmethod
    def get_instrument_by_name(cls, pair_name):
        d = Instrument.get_instruments_dict()
        return (d[pair_name] if pair_name in d else None)
    
    
if __name__=='__main__':
    print(Instrument.get_instruments_df())