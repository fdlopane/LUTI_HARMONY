"""
quantretailmodel.py
Build a retail model for QUANT
"""

import pandas as pd

from HARMONY_LUTI_OXF.quantlhmodel import QUANTLHModel

class QUANTRetailModel(QUANTLHModel):
    """
    constructor
    @param n number of residential zones
    @param m number of retail zones
    """
    def __init__(self,m,n):
        #constructor
        super().__init__(m,n)

    ################################################################################

    """
    loadGeolytixData
    @param filename Name of file to load - this is the Geolytix restricted access data with
    the floorspace and retail data
    @returns DataFrame containing [key,zonei,east,north] and [zonei,Modelled turnover annual]
    """
    @staticmethod
    def loadGeolytixData(filename):
        missing_values = ['-', 'n/a', 'na', '--', ' -   ']
        df = pd.read_csv(filename,usecols=['id','fascia','modelled sq ft','Modelled turnover annual','bng_e','bng_n'], na_values=missing_values)
        df.dropna(axis=0,inplace=True)
        df.reset_index(drop=True,inplace=True) # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        dfzones = pd.DataFrame({'id':df.id,'zonei':df.index,'east':df.bng_e,'north':df.bng_n})
        dfattractors = pd.DataFrame({'zonei':df.index,'Modelled turnover annual':df['Modelled turnover annual']}) # could also used floorspace
        return dfzones, dfattractors