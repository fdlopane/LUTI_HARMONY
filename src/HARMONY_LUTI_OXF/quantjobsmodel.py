"""
quantjobsmodel.py
Build a journey to work model for QUANT
"""

import pandas as pd
import os
from HARMONY_LUTI_OXF.globals import *

from HARMONY_LUTI_OXF.quantlhmodel import QUANTLHModel
from HARMONY_LUTI_OXF.utils import loadQUANTMatrix, loadMatrix, saveMatrix

class QUANTJobsModel(QUANTLHModel):
    """
    constructor
    @param n number of residential zones
    @param m number of retail zones
    """

    def __init__(self, m, n):
        # constructor
        super().__init__(m, n)

    """
    loadEmploymentData
    - Load jobs by zone: df with columns = [,zonei,employment]
    - Load floorspace by zone: df with columns = [,zonei,floorspace] 
    return dfzones, dfattractors (they are DataFrames)
    """
    @staticmethod
    def loadEmploymentData_HHAttractiveness(filename_pop, filename_attractiveness, Scenario):
        missing_values = ['-', 'n/a', 'na', '--', ' -   ']

        # load population data for MSOA residential zones
        df = pd.read_csv(filename_pop,
                         usecols=['geography code', 'Age: All categories: Age; measures: Value'],
                         na_values=missing_values)  # import population data for England and Wales

        df.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!

        # Rename columns:
        df.rename(columns={'geography code': 'zonei',
                           'Age: All categories: Age; measures: Value': 'Population_tot'}, inplace=True)

        if Scenario == '2011':
            df2 = pd.read_csv(filename_attractiveness, encoding='latin-1') # encoding added to solve: UnicodeDecodeError
        elif Scenario == 'NewHousingDev_2019':
            df2 = pd.read_csv(filename_attractiveness, encoding='latin-1') # encoding added to solve: UnicodeDecodeError
        elif Scenario == 'NewHousingDev_2030':
            df2 = pd.read_csv(filename_attractiveness, encoding='latin-1') # encoding added to solve: UnicodeDecodeError

        df2.rename(columns={'MSOA_Code': 'zonei', 'N_of_Dwellings': 'N_of_Dwellings'}, inplace=True)

        # Convert N_of_Dwellings to float
        df2['N_of_Dwellings'] = df2.N_of_Dwellings.astype(float)

        dfzones = pd.DataFrame({'zonei': df.zonei, 'Population_tot': df.Population_tot})
        dfattractors = pd.DataFrame({'zonei': df2.zonei, 'N_of_Dwellings': df2.N_of_Dwellings})
        return dfzones, dfattractors

    @staticmethod
    def loadEmploymentData_JobAttractiveness(filename_jobs, filename_attractiveness): # Attractiveness: offices floorspace
        missing_values = ['-', 'n/a', 'na', '--', ' -   ']

        df = pd.read_csv(filename_jobs,
                         usecols=['geography code','Industry: All categories: Industry; measures: Value'], # select the columms that I need from file
                         na_values=missing_values)
        #df.dropna(axis=0, inplace=True) # I think I want to keep all the MSOAs even if they have null alues - in case change them

        df.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!

        # Rename columns:
        df.rename(columns={'geography code': 'zonei', 'Industry: All categories: Industry; measures: Value': 'employment_tot'}, inplace=True)

        df2 = pd.read_csv(filename_attractiveness,
                          encoding='latin-1', # added to solve: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf4 in position 25: invalid continuation byte
                         usecols=['Geography', 'area_code', 'Area_Name', 'Floorspace_2018-19_Total'])  # select the columms that I need from file
                         #na_values=missing_values)
        #df2.dropna(axis=0, inplace=True) # I think I want to keep all the MSOAs even if they have null alues - in case change them

        # filter out MSOA rows in jobs_floorspace:
        df2 = df2[df2.Geography == 'MSOA']

        df2.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!

        df2.rename(columns={'area_code': 'zonei', 'Floorspace_2018-19_Total': 'floorspace'}, inplace=True)

        for m in missing_values:
            df2['floorspace'] = df2.floorspace.str.replace(m,'1.0') # Replace missing values with the minimum value of df2: 1.0

        # Convert floorspace to float
        df2['floorspace'] = df2.floorspace.str.replace(',', '').astype(float)

        #df2.fillna(1.0)  # Replace nan with the minimum value of df2 - 1.0

        dfzones = pd.DataFrame({'zonei': df.zonei, 'employment_tot': df.employment_tot})
        dfattractors = pd.DataFrame({'zonei': df2.zonei, 'floorspace': df2.floorspace})
        return dfzones, dfattractors

    @staticmethod
    def loadObsData():
        # load observed trips for each mode:
        SObs_0 = loadMatrix(os.path.join(modelRunsDir, SObsRoadFilename_OXF))
        SObs_1 = loadMatrix(os.path.join(modelRunsDir, SObsBusFilename_OXF))
        SObs_2 = loadMatrix(os.path.join(modelRunsDir, SObsRailFilename_OXF))

        return SObs_0, SObs_1, SObs_2