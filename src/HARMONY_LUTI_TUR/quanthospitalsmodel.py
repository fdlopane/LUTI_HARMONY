"""
quanthospitalsmodel.py
Build a travel to hospitals model for QUANT
"""

import pandas as pd

from HARMONY_LUTI_TUR.quantlhmodel import QUANTLHModel

class QUANTHospitalsModel(QUANTLHModel):
    """
    constructor
    @param n number of residential zones
    @param m number of hospital zones
    """
    def __init__(self,m,n):
        #constructor
        super().__init__(m,n)

    """
    loadHospitalsData
    @param filename Name of file to load - this is the NHS dataset containing the hospital locations and numbers of beds
    @returns DataFrame containing [key,zonei,east,north] and [zonei,beds]
    """
    @staticmethod
    def loadHospitalsData(filename):
        missing_values = ['-', 'n/a', 'na', '--', ' -   ']
        df = pd.read_csv(filename,usecols=['code','name','beds','lat','long'], na_values=missing_values)
        df.dropna(axis=0,inplace=True)
        df.reset_index(drop=True,inplace=True) # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        dfzones = pd.DataFrame({'id':df['code'],'name':df['name'],'zonei':df.index,'latitude':df['lat'],'longitude':df['long']})
        dfattractors = pd.DataFrame({'zonei':df.index,'Number_of_beds':df['beds']})
        return dfzones, dfattractors