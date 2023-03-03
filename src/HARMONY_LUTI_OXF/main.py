"""
HARMONY Land-Use Transport-Interaction Model - Oxfordshire case study
main.py

17 September 2020
Author: Fulvio D. Lopane, Centre for Advanced Spatial Analysis, University College London
https://www.casa.ucl.ac.uk
"""
import os
import time
import pandas as pd
import numpy as np
from geojson import dump

from HARMONY_LUTI_OXF.globals import *
from HARMONY_LUTI_OXF.maps import *
from HARMONY_LUTI_OXF.utils import loadQUANTMatrix, loadMatrix, saveMatrix
from HARMONY_LUTI_OXF.databuilder import ensureFile, changeGeography
# from HARMONY_LUTI_OXF.databuilder import geolytixRegression, geolytixOpenDataRegression, geocodeGeolytix, computeGeolytixCosts
from HARMONY_LUTI_OXF.databuilder import buildSchoolsPopulationTableEnglandWales, buildSchoolsPopulationTableScotland
from HARMONY_LUTI_OXF.databuilder import buildTotalPopulationTable
from HARMONY_LUTI_OXF.quantretailmodel import QUANTRetailModel
from HARMONY_LUTI_OXF.quantschoolsmodel import QUANTSchoolsModel
from HARMONY_LUTI_OXF.quanthospitalsmodel import QUANTHospitalsModel
from HARMONY_LUTI_OXF.quantjobsmodel import QUANTJobsModel
from HARMONY_LUTI_OXF.costs import costMSOAToPoint_3modes
from HARMONY_LUTI_OXF.analytics import graphProbabilities, flowArrowsGeoJSON


def start_main(inputs, outputs):
    ################################################################################
    # Initialisation                                                               #
    ################################################################################

    # NOTE: this section provides the base data for the models that come later. This
    # will only be run on the first run of the program to assemble all the tables
    # required from the original sources. After that, if the file exists in the
    # directory, then nothing new is created and this section is effectively
    # skipped, up until the model run section at the end.

    # make a model-runs dir if we need it
    if not os.path.exists(modelRunsDir):
        os.makedirs(modelRunsDir)

    # Downloads first:

    # this will get the QUANT travel times matrices
    ensureFile(os.path.join(modelRunsDir,QUANTCijRoadMinFilename),url_QUANTCijRoadMinFilename)
    ensureFile(os.path.join(modelRunsDir,ZoneCodesFilename),url_QUANT_ZoneCodes) # zone code lookup that goes with it
    ensureFile(os.path.join(modelRunsDir,QUANTCijRoadCentroidsFilename),url_QUANT_RoadCentroids) # road centroids
    ensureFile(os.path.join(modelRunsDir,QUANTCijBusCentroidsFilename),url_QUANT_BusCentroids) # bus centroids
    ensureFile(os.path.join(modelRunsDir,QUANTCijRailCentroidsFilename),url_QUANT_RailCentroids) # rail centroids

    ################################################################################
    # File creation

    # This is the open data version of the Geolytix restricted data regression - uses linear regression params derived from the restricted data that we can't release
    if not os.path.isfile(inputs["DataOpenGeolytixRegression"]):
        print("ERROR: You need the Open Data Geolytix Regression input file to run the retail model")
        # geolytixOpenDataRegression(inputs["DataGeolytixRetailpoints"],inputs["DataOpenGeolytixRegression"]) # Not open data

    # databuilder.py - build schools population table if needed - requires QS103 AND QS103SC on DZ2001 for Scotland
    if not os.path.isfile(data_schoolagepopulation_englandwales):
        buildSchoolsPopulationTableEnglandWales(inputs["DataCensusQS103"])

    if not os.path.isfile(data_schoolagepopulation_scotland):
        buildSchoolsPopulationTableScotland(inputs["DataCensusQS103SC"], inputs["LookupDZ2001toIZ2001"])

    # now join England/Wales and Scotland files together in case we wanted an EWS model:
    if not os.path.isfile(data_schoolagepopulation):
        df1 = pd.read_csv(data_schoolagepopulation_englandwales)
        df1.columns=['rowid','msoaiz','count_primary','count_secondary']
        df2 = pd.read_csv(data_schoolagepopulation_scotland)
        df2.columns=['msoaiz','count_primary','count_secondary']
        df3 = df1.append(df2)
        df3.to_csv(data_schoolagepopulation)

    # same thing for the total population of England, Scotland and Wales - needed for hospitals
    # now join total population for England/Wales and Scotland files together
    if not os.path.isfile(data_totalpopulation):
        buildTotalPopulationTable(inputs["DataCensusQS103"], inputs["DataCensusQS103SC"], inputs["LookupDZ2001toIZ2001"])

    ################################################################################
    # Model run section
    ################################################################################
    print()
    print("Importing QUANT cij matrices")

    OXF_MSOA_df =  pd.read_csv(inputs["OxfMsoaFile"], usecols=["msoa11cd"]) # import file with OXF MSOA codes
    OXF_MSOA_df.columns = ['areakey'] # rename column "msoa11cd" to "areakey"
    OXF_MSOA_list = OXF_MSOA_df['areakey'].tolist()

    # load zone codes lookup file to convert MSOA codes into zone i indexes for the model
    zonecodes_EWS = pd.read_csv(os.path.join(modelRunsDir,ZoneCodesFilename))
    zonecodes_EWS.set_index('areakey')
    zonecodes_EWS_list = zonecodes_EWS['areakey'].tolist()

    #_____________________________________________________________________________________
    # IMPORT cij QUANT matrices
    # ROADS cij
    print()
    print("Importing QUANT roads cij for Oxfordshire")

    if not os.path.isfile(os.path.join(modelRunsDir,QUANTCijRoadMinFilename_OXF)):
        # load cost matrix, time in minutes between MSOA zones for roads:
        cij_road_EWS = loadQUANTMatrix(inputs["QUANTCijRoadMinFilename"])
        cij_road_EWS_df = pd.DataFrame(cij_road_EWS, index=zonecodes_EWS_list, columns=zonecodes_EWS_list) # turn the numpy array into a pd dataframe, (index and columns: MSOA codes)
        cij_road_OXF_df = cij_road_EWS_df[OXF_MSOA_list]  # Create OXF df filtering EWS columns
        cij_road_OXF_df = cij_road_OXF_df.loc[OXF_MSOA_list]  # Filter rows
        cij_road_OXF = cij_road_OXF_df.to_numpy()  # numpy matrix for OXF (same format as utils loadQUANTMatrix)
        cij_road_OXF[cij_road_OXF < 1] = 1  # lower limit of 1 minute links
        saveMatrix(cij_road_OXF, os.path.join(modelRunsDir,QUANTCijRoadMinFilename_OXF))
        # save as csv file
        np.savetxt(os.path.join(modelRunsDir, "cij_road_OXF.csv"), cij_road_OXF, delimiter=",")
    else:
        cij_road_OXF = loadMatrix(os.path.join(modelRunsDir,QUANTCijRoadMinFilename_OXF))
        print('cij roads shape: ', cij_road_OXF.shape)

    # Export cij matrices for checking
    # np.savetxt(os.path.join(modelRunsDir,'debug_cij_roads.csv'), cij_road_OXF, delimiter=',', fmt='%i')

    #_____________________________________________________________________________________
    # BUS & FERRIES cij
    print()
    print("Importing QUANT bus cij for Oxfordshire")

    if not os.path.isfile(os.path.join(modelRunsDir,QUANTCijBusMinFilename_OXF)):
        # load cost matrix, time in minutes between MSOA zones for bus and ferries network:
        cij_bus_EWS = loadQUANTMatrix(inputs["QUANTCijBusMinFilename"])
        cij_bus_EWS_df = pd.DataFrame(cij_bus_EWS, index=zonecodes_EWS_list, columns=zonecodes_EWS_list) # turn the numpy array into a pd dataframe, (index and columns: MSOA codes)
        cij_bus_OXF_df = cij_bus_EWS_df[OXF_MSOA_list]  # Create OXF df filtering EWS columns
        cij_bus_OXF_df = cij_bus_OXF_df.loc[OXF_MSOA_list]  # Filter rows
        cij_bus_OXF = cij_bus_OXF_df.to_numpy()  # numpy matrix for OXF (same format as utils loadQUANTMatrix)
        cij_bus_OXF[cij_bus_OXF < 1] = 1  # lower limit of 1 minute links
        saveMatrix(cij_bus_OXF, os.path.join(modelRunsDir, QUANTCijBusMinFilename_OXF))
        # save as csv file
        np.savetxt(os.path.join(modelRunsDir, "cij_bus_OXF.csv"), cij_bus_OXF, delimiter=",")
    else:
        cij_bus_OXF = loadMatrix(os.path.join(modelRunsDir,QUANTCijBusMinFilename_OXF))
        print('cij bus shape: ', cij_bus_OXF.shape)

    # Export cij matrices for checking
    # np.savetxt(os.path.join(modelRunsDir,'debug_cij_bus.csv'), cij_bus_OXF, delimiter=',', fmt='%i')

    #_____________________________________________________________________________________
    # RAILWAYS cij
    print()
    print("Importing QUANT rail cij for Oxfordshire")

    if not os.path.isfile(os.path.join(modelRunsDir,QUANTCijRailMinFilename_OXF)):
        # load cost matrix, time in minutes between MSOA zones for railways:
        cij_rail_EWS = loadQUANTMatrix(inputs["QUANTCijRailMinFilename"])
        cij_rail_EWS_df = pd.DataFrame(cij_rail_EWS, index=zonecodes_EWS_list, columns=zonecodes_EWS_list) # turn the numpy array into a pd dataframe, (index and columns: MSOA codes)
        cij_rail_OXF_df = cij_rail_EWS_df[OXF_MSOA_list]  # Create OXF df filtering EWS columns
        cij_rail_OXF_df = cij_rail_OXF_df.loc[OXF_MSOA_list]  # Filter rows
        cij_rail_OXF = cij_rail_OXF_df.to_numpy()  # numpy matrix for OXF (same format as utils loadQUANTMatrix)
        cij_rail_OXF[cij_rail_OXF < 1] = 1  # lower limit of 1 minute links
        saveMatrix(cij_rail_OXF, os.path.join(modelRunsDir, QUANTCijRailMinFilename_OXF))
        # save as csv file
        np.savetxt(os.path.join(modelRunsDir, "cij_rail_OXF.csv"), cij_rail_OXF, delimiter=",")
    else:
        cij_rail_OXF = loadMatrix(os.path.join(modelRunsDir,QUANTCijRailMinFilename_OXF))
        print('cij rail shape: ', cij_rail_OXF.shape)

    # Export cij matrices for checking
    # np.savetxt(os.path.join(modelRunsDir,'debug_cij_rail.csv'), cij_rail_OXF, delimiter=',', fmt='%i')

    print()
    print("Importing QUANT cij matrices completed.")
    print()
    #_____________________________________________________________________________________

    # IMPORT SObs QUANT matrices: observed trips
    print("Importing SObs matrices")

    # SObs ROADS
    print()
    print("Importing SObs for roads for Oxfordshire")

    if not os.path.isfile(os.path.join(modelRunsDir,SObsRoadFilename_OXF)):
        # load observed trips matrix for roads:
        SObs_road_EWS = loadQUANTMatrix(inputs["SObsRoadFilename"])
        SObs_road_EWS_df = pd.DataFrame(SObs_road_EWS, index=zonecodes_EWS_list, columns=zonecodes_EWS_list) # turn the numpy array into a pd dataframe, (index and columns: MSOA codes)
        SObs_road_OXF_df = SObs_road_EWS_df[OXF_MSOA_list]  # Create OXF df filtering EWS columns
        SObs_road_OXF_df = SObs_road_OXF_df.loc[OXF_MSOA_list]  # Filter rows
        SObs_road_OXF = SObs_road_OXF_df.to_numpy()  # numpy matrix for OXF (same format as utils loadQUANTMatrix)
        saveMatrix(SObs_road_OXF, os.path.join(modelRunsDir, SObsRoadFilename_OXF))
    # else:
    #     SObs_road_OXF = loadMatrix(os.path.join(modelRunsDir,SObsRoadFilename_OXF))
    #     print('Sobs road shape: ', SObs_road_OXF.shape)
    #_____________________________________________________________________________________

    # SObs BUS & FERRIES
    print()
    print("Importing SObs for bus & ferries for Oxfordshire")

    if not os.path.isfile(os.path.join(modelRunsDir,SObsBusFilename_OXF)):
        # load observed trips matrix for bus and ferries:
        SObs_bus_EWS = loadQUANTMatrix(inputs["SObsBusFilename"])
        SObs_bus_EWS_df = pd.DataFrame(SObs_bus_EWS, index=zonecodes_EWS_list, columns=zonecodes_EWS_list)  # turn the numpy array into a pd dataframe, (index and columns: MSOA codes)
        SObs_bus_OXF_df = SObs_bus_EWS_df[OXF_MSOA_list]  # Create OXF df filtering EWS columns
        SObs_bus_OXF_df = SObs_bus_OXF_df.loc[OXF_MSOA_list]  # Filter rows
        SObs_bus_OXF = SObs_bus_OXF_df.to_numpy()  # numpy matrix for OXF (same format as utils loadQUANTMatrix)
        saveMatrix(SObs_bus_OXF, os.path.join(modelRunsDir,SObsBusFilename_OXF))
    # else:
    #     SObs_bus_OXF = loadMatrix(os.path.join(modelRunsDir,SObsBusFilename_OXF))
    #     print('Sobs bus shape: ', SObs_bus_OXF.shape)
    #_____________________________________________________________________________________

    # SObs RAIL
    print()
    print("Importing SObs for rail for Oxfordshire")

    if not os.path.isfile(os.path.join(modelRunsDir,SObsRailFilename_OXF)):
        # load observed trips matrix for rails:
        SObs_rail_EWS = loadQUANTMatrix(inputs["SObsRailFilename"])
        SObs_rail_EWS_df = pd.DataFrame(SObs_rail_EWS, index=zonecodes_EWS_list, columns=zonecodes_EWS_list)  # turn the numpy array into a pd dataframe, (index and columns: MSOA codes)
        SObs_rail_OXF_df = SObs_rail_EWS_df[OXF_MSOA_list]  # Create OXF df filtering EWS columns
        SObs_rail_OXF_df = SObs_rail_OXF_df.loc[OXF_MSOA_list]  # Filter rows
        SObs_rail_OXF = SObs_rail_OXF_df.to_numpy()  # numpy matrix for OXF (same format as utils loadQUANTMatrix)
        saveMatrix(SObs_rail_OXF, os.path.join(modelRunsDir,SObsRailFilename_OXF))
    # else:
    #     SObs_rail_OXF = loadMatrix(os.path.join(modelRunsDir,SObsRailFilename_OXF))
    #     print('Sobs bus shape: ', SObs_rail_OXF.shape)
    #     print()
    #_____________________________________________________________________________________


    # now run the relevant models to produce the outputs
    runNewHousingDev(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, inputs, outputs)

    # Maps creation:

    # Population maps:
    population_map_creation(inputs, outputs)

    # Flows maps:
    # THIS FEATURE IS TURNED OFF - long run time - only for HQ flows visualisation
    create_flow_maps = False
    if create_flow_maps:
        flows_output_keys = ["JobsTijRoads2019", "JobsTijBus2019", "JobsTijRoads2030", "JobsTijBus2030"]
        flows_map_creation(inputs, outputs, flows_output_keys)

    # Low quality (light .png) maps:
    # flows_map_creation_light(inputs, outputs, "JobsTijRoads2019", inputs["RoadNetworkShapefile"]) # 2019 Jobs Roads
    # flows_map_creation_light(inputs, outputs, "JobsTijBus2019", inputs["RoadNetworkShapefile"]) # 2019 Jobs Bus
    # flows_map_creation_light(inputs, outputs, "JobsTijRoads2030", inputs["RoadNetworkShapefile"])  # 2030 Jobs Roads
    # flows_map_creation_light(inputs, outputs, "JobsTijBus2030", inputs["RoadNetworkShapefile"])  # 2030 Jobs Bus

################################################################################
# End initialisation
################################################################################

################################################################################
# New housing development scenario                                             #
################################################################################
"""
New housing development scenario:
first run the Journey to Work model (for calibration)
then change the population reading from the new housing development table
and make predictions based on the new population 
"""

def runNewHousingDev(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, inputs, outputs):
    # First run the base model to calibrate it with 2011 observed trip data:
    # Run Journey to work model:
    beta_2011, DjPred_JtW_2011 = runJourneyToWorkModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, inputs, outputs)

    # HARMONY new housing development scenario:
    # base year: 2019, projection year: 2030
    # First, create the new population files: updated population per MSOA zone
    # (starting from table with singular new housing locations with number of new houses)
    # 2019 (base year): use census (2011) + new houses 2011_2020
    # 2030 (proj year): use census (2011) + new houses 2026_2031 (it takes into account also the previous years)
    # Then, run the models with this new populations

    # Create pop file for 2019 and 2030:
    # read csv from New_housing_dev_table and ue columns: "MSOA" and "2011_2020"
    NewHousingDev_table_2019 = pd.read_csv(inputs["OXFNewHousingDev"], usecols=['MSOA', '2011_2020']) # for 2019 read the entry up to year 2020
    NewHousingDev_table_2030 = pd.read_csv(inputs["OXFNewHousingDev"], usecols=['MSOA', '2026_2031']) # for 2030 read the entry up to year 2031

    # Group by MSOA and calculate population from number of houses
    Av_HH_size = 2.4  # The average household size in the UK is 2.4, from ONS (Families and households in the UK: 2020)

    NewHousingPop_2019 = NewHousingDev_table_2019.groupby(['MSOA'], as_index=False).sum()
    NewHousingPop_2019.rename(columns={'2011_2020':'Newhouses_2011_2020'}, inplace=True)
    NewHousingPop_2019['NewPop_2011_2019'] = NewHousingPop_2019['Newhouses_2011_2020'] * Av_HH_size # it's ok that it's not an integer as the final result will be a float anyway
    NewHousingPop_2019 = NewHousingPop_2019[['MSOA', 'NewPop_2011_2019']] # drop the n of houses column

    NewHousingPop_2030 = NewHousingDev_table_2030.groupby(['MSOA'], as_index=False).sum()
    NewHousingPop_2030.rename(columns={'2026_2031': 'Newhouses_2011_2031'}, inplace=True)
    NewHousingPop_2030['NewPop_2011_2030'] = NewHousingPop_2030['Newhouses_2011_2031'] * Av_HH_size  # it's ok that it's not an integer as the final result will be a float anyway
    NewHousingPop_2030 = NewHousingPop_2030[['MSOA', 'NewPop_2011_2030']] # drop the n of houses column

    # Now run the JtW model with 2011 beta and 2019 pop (without calibration this time)

    beta_2011, DjPred_JtW_2019 = runJourneyToWorkModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, inputs, outputs, 'NewHousingDev_2019', NewHousingPop_2019, beta_2011)
    beta_2011, DjPred_JtW_2030 = runJourneyToWorkModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, inputs, outputs, 'NewHousingDev_2030', NewHousingPop_2030, beta_2011)

    # NewHousingDev_2019
    # Run Population Retail model:
    runPopulationRetailModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_2011, inputs, outputs, NewHousingPop_2019, Scenario='NewHousingDev_2019')
    # Run Schools model:
    runSchoolsModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_2011, NewHousingPop_2019, inputs, outputs, Scenario='NewHousingDev_2019')
    # Run Hospitals model:
    runHospitalsModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_2011, NewHousingPop_2019, inputs, outputs, Scenario='NewHousingDev_2019')

    # NewHousingDev_2030
    # Run Population Retail model:
    runPopulationRetailModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_2011, inputs, outputs, NewHousingPop_2030, Scenario='NewHousingDev_2030')
    # Run Schools model:
    runSchoolsModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_2011, NewHousingPop_2030, inputs, outputs, Scenario='NewHousingDev_2030')
    # Run Hospitals model:
    runHospitalsModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_2011, NewHousingPop_2030, inputs, outputs, Scenario='NewHousingDev_2030')


# What follows from here are the different model run functions for retail, schools, hospitals and journey to work.

# Areas abbreviations for models' data:
# EW = England + Wales
# EWS = England + Wales + Scotland
# OXF = Oxfordshire

################################################################################
# Journey to work Model                                                        #
################################################################################

"""
runJourneyToWorkModel
Origins: workplaces, Destinations: households' population
"""
# Journey to work model with households (HH) floorspace as attractor
def runJourneyToWorkModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, inputs, outputs, Scenario='2011', Scenario_pop_table=None, Beta_calibrated=None):
    print("Running Journey to Work ", Scenario, " model.")
    start = time.perf_counter()
    # Singly constrained model:
    # We conserve the number of jobs and predict the population residing in MSOA zones
    # journeys to work generated by jobs
    # Origins: workplaces
    # Destinations: MSOAs households
    # Attractor: number of dwellings

    """
                    Journey to work    |   Retail model
     Origins:       workplaces         |   households
     Destinations:  households         |   supermarkets
     conserved:     jobs               |   income
     predicted:     population @ MSOA  |   expenditure @ supermarkets
     attractor:     N of dwellings     |   supermarket floorspace
    """

    # load jobs data for MSOA residential zones
    dfEi = pd.read_csv(inputs["Employment2011GB"], usecols=['geography code','Industry: All categories: Industry; measures: Value'], index_col='geography code') # select the columns that I need from file

    # Rename columns:
    dfEi.rename(columns={'geography code': 'msoa', 'Industry: All categories: Industry; measures: Value': 'employment_tot'}, inplace=True)

    # Filter out Oxfordshire from EW dataset:
    dfEi = dfEi.loc[OXF_MSOA_list]  # Filter rows

    dfEi['msoa'] = dfEi.index # turn the index (i.e. MSOA codes) back into a columm

    dfEi.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!

    # drop columns:
    dfEi = dfEi[['msoa','employment_tot']]
    jobs = dfEi.join(other=zonecodes_EWS.set_index('areakey'), on='msoa')  # this codes dfEi by zonei
    # jobs.to_csv(data_jobs_employment) # save file to csv in model-runs directory

    # load the households data file and make an attraction vector from the number of dwellings
    # Filter out Oxfrodshire from EW census and floorspace data sets:

    if not os.path.isfile(data_census_QS103_OXF):
        data_census_QS103_EW_df = pd.read_csv(inputs["DataCensusQS103"], index_col='geography code')
        data_census_QS103_OXF_df = data_census_QS103_EW_df.loc[OXF_MSOA_list]  # Filter rows
        data_census_QS103_OXF_df['geography code'] = data_census_QS103_OXF_df.index # turn the index (i.e. MSOA codes) back into a columm
        data_census_QS103_OXF_df.reset_index(drop=True, inplace=True)
        data_census_QS103_OXF_df.to_csv(data_census_QS103_OXF)

    HHZones, HHAttractors = QUANTJobsModel.loadEmploymentData_HHAttractiveness(data_census_QS103_OXF, inputs["DwellingsOxf"], Scenario)

    # if we are running a scenario, update the HHZones with the new population
    if Scenario == '2011':
        HHZones.to_csv(data_HH_zones_2011)  # save file to csv in model-runs directory
        HHAttractors.to_csv(data_HH_attractors_2011)  # save file to csv in model-runs directory

    elif Scenario == 'NewHousingDev_2019':
        dfEi = pd.read_csv(inputs["Employment2019"])  # select the columns that I need from file
        dfEi.rename(columns={'2019': 'employment_tot'}, inplace=True)
        jobs = dfEi.join(other=zonecodes_EWS.set_index('areakey'), on='msoa')  # this codes dfEi by zonei

        # Rename Scenario_pop_table's columns:
        Scenario_pop_table.rename(columns={'MSOA':'zonei'}, inplace=True)

        # Update the population with the 2019 projection
        HHZones = HHZones.join(Scenario_pop_table.set_index('zonei'), on=['zonei'])
        HHZones['NewPop_2011_2019'] = HHZones['NewPop_2011_2019'].fillna(0)  # Replace NaN with 0
        HHZones['Population_tot'] = HHZones['Population_tot'] + HHZones['NewPop_2011_2019'] # Sum the additional pop to previous one
        HHZones = HHZones[['zonei', 'Population_tot']]

        # Update the number of dwellings with the 2019 projection
        NewHousingDev_table_2019 = pd.read_csv(inputs["OXFNewHousingDev"], usecols=['MSOA', '2011_2020'])  # for 2019 read the entry up to year 2020
        NewHousingDev_table_2019.rename(columns={'MSOA': 'zonei', '2011_2020': 'Newhouses_2011_2020'}, inplace=True) # Rename columns
        NewHousingDev_table_2019 = NewHousingDev_table_2019.groupby(['zonei'], as_index=False).sum() # Sum the new houses in the same zone

        HHAttractors = HHAttractors.join(NewHousingDev_table_2019.set_index('zonei'), on=['zonei']) # Join the Attractors df with the new houses df
        HHAttractors['Newhouses_2011_2020'] = HHAttractors['Newhouses_2011_2020'].fillna(0) # Replace NaN with 0
        HHAttractors['N_of_Dwellings'] = HHAttractors['N_of_Dwellings'] + HHAttractors['Newhouses_2011_2020'] # Add the new houses to existing df
        HHAttractors = HHAttractors[['zonei', 'N_of_Dwellings']]

        HHZones.to_csv(data_HH_zones_2019)  # save file to csv in model-runs directory
        HHAttractors.to_csv(data_HH_attractors_2019)  # save file to csv in model-runs directory

    elif Scenario == 'NewHousingDev_2030':
        dfEi = pd.read_csv(inputs["Employment2030"])  # select the columns that I need from file
        dfEi.rename(columns={'2030': 'employment_tot'}, inplace=True)
        jobs = dfEi.join(other=zonecodes_EWS.set_index('areakey'), on='msoa')  # this codes dfEi by zonei

        # Rename Scenario_pop_table's columns:
        Scenario_pop_table.rename(columns={'MSOA':'zonei'}, inplace=True)

        # Update the population with the 2030 projection
        HHZones = HHZones.join(Scenario_pop_table.set_index('zonei'), on=['zonei'])
        HHZones['NewPop_2011_2030'] = HHZones['NewPop_2011_2030'].fillna(0)  # Replace NaN with 0
        HHZones['Population_tot'] = HHZones['Population_tot'] + HHZones['NewPop_2011_2030']
        HHZones = HHZones[['zonei', 'Population_tot']]

        # Update the number of dwellings with the 2030 projection
        NewHousingDev_table_2030 = pd.read_csv(inputs["OXFNewHousingDev"], usecols=['MSOA', '2026_2031'])  # for 2030 read the entry up to year 2031
        NewHousingDev_table_2030.rename(columns={'MSOA': 'zonei', '2026_2031': 'Newhouses_2026_2031'}, inplace=True)
        NewHousingDev_table_2030 = NewHousingDev_table_2030.groupby(['zonei'], as_index=False).sum()  # Sum the new houses in the same zone

        HHAttractors = HHAttractors.join(NewHousingDev_table_2030.set_index('zonei'), on=['zonei'])  # Join the Attractors df with the new houses df
        HHAttractors['Newhouses_2026_2031'] = HHAttractors['Newhouses_2026_2031'].fillna(0)  # Replace NaN with 0
        HHAttractors['N_of_Dwellings'] = HHAttractors['N_of_Dwellings'] + HHAttractors['Newhouses_2026_2031']
        HHAttractors = HHAttractors[['zonei', 'N_of_Dwellings']]

        HHZones.to_csv(data_HH_zones_2030)  # save file to csv in model-runs directory
        HHAttractors.to_csv(data_HH_attractors_2030)  # save file to csv in model-runs directory

    # Now run the model with or without calibration according to the scenario:
    if Scenario == '2011':
        # Load observed data for model calibration:
        SObs_road, SObs_bus, SObs_rail = QUANTJobsModel.loadObsData()

        # Use cij as cost matrix (MSOA to MSOA)
        m, n = cij_road_OXF.shape
        model = QUANTJobsModel(m, n)
        model.setObsMatrix(SObs_road, SObs_bus, SObs_rail)
        model.setAttractorsAj(HHAttractors, 'zonei', 'N_of_Dwellings')
        model.setPopulationEi(jobs, 'zonei', 'employment_tot')
        model.setCostMatrixCij(cij_road_OXF, cij_bus_OXF, cij_rail_OXF)

        Tij, beta_k, cbar_k = model.run3modes() # run the model with 3 modes + calibration

        # Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_road = Tij[0].sum(axis=1)
        Ji_road = Calculate_Job_Accessibility(DjPred_road, cij_road_OXF)

        DjPred_bus = Tij[1].sum(axis=1)
        Ji_bus = Calculate_Job_Accessibility(DjPred_bus, cij_bus_OXF)

        DjPred_rail = Tij[2].sum(axis=1)
        Ji_rail = Calculate_Job_Accessibility(DjPred_rail, cij_rail_OXF)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame( {'areakey': OXF_MSOA_list, 'JAcar11': Ji_road, 'JAbus11': Ji_bus, 'JArail11': Ji_rail})
        Jobs_accessibility_df.to_csv(data_jobs_accessibility_2011)

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.

        OiPred_road = Tij[0].sum(axis=0)
        Hi_road = Calculate_Housing_Accessibility(OiPred_road, cij_road_OXF)

        OiPred_bus = Tij[1].sum(axis=0)
        Hi_bus = Calculate_Housing_Accessibility(OiPred_bus, cij_bus_OXF)

        OiPred_rail = Tij[2].sum(axis=0)
        Hi_rail = Calculate_Housing_Accessibility(OiPred_rail, cij_rail_OXF)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'areakey': OXF_MSOA_list, 'HAcar11': Hi_road, 'HAbus11': Hi_bus, 'HArail11': Hi_rail})
        Housing_accessibility_df.to_csv(data_housing_accessibility_2011)

        # Create a Oi Dj table
        jobs['DjPred_Car_11'] = Tij[0].sum(axis=1)
        jobs['DjPred_Bus_11'] = Tij[1].sum(axis=1)
        jobs['DjPred_Rail_11'] = Tij[2].sum(axis=1)
        jobs['DjPred_Tot_11'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1) + Tij[2].sum(axis=1)
        jobs['OiPred_Car_11'] = Tij[0].sum(axis=0)
        jobs['OiPred_Bus_11'] = Tij[1].sum(axis=0)
        jobs['OiPred_Rail_11'] = Tij[2].sum(axis=0)
        jobs['OiPred_Tot_11'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis=0)
        jobs['Job_accessibility_roads'] = Jobs_accessibility_df['JAcar11']
        jobs['Jobs_accessibility_bus'] = Jobs_accessibility_df['JAbus11']
        jobs['Jobs_accessibility_rail'] = Jobs_accessibility_df['JArail11']
        jobs['Housing_accessibility_roads'] = Housing_accessibility_df['HAcar11']
        jobs['Housing_accessibility_bus'] = Housing_accessibility_df['HAbus11']
        jobs['Housing_accessibility_rail'] = Housing_accessibility_df['HArail11']
        jobs.to_csv(Jobs_DjOi_2011)

        # Save output matrices
        print("Saving output matrices...")

        # Probabilities:
        np.savetxt(data_jobs_probTij_roads_2011_csv, jobs_probTij[0], delimiter=",")
        np.savetxt(data_jobs_probTij_bus_2011_csv, jobs_probTij[1], delimiter=",")
        np.savetxt(data_jobs_probTij_rail_2011_csv, jobs_probTij[2], delimiter=",")

        # People flows
        np.savetxt(data_jobs_Tij_roads_2011_csv, Tij[0], delimiter=",")
        np.savetxt(data_jobs_Tij_bus_2011_csv, Tij[1], delimiter=",")
        np.savetxt(data_jobs_Tij_rail_2011_csv, Tij[2], delimiter=",")

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and GB grid indexes
        flow_zonecodes = pd.read_csv(inputs["ZonesCoordinates"])
        flow_car = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(os.path.join(modelRunsDir, 'flows_2011_car.geojson'), 'w') as f:
            dump(flow_car, f)
        flow_bus = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(os.path.join(modelRunsDir, 'flows_2011_bus.geojson'), 'w') as f:
            dump(flow_bus, f)
        flow_rail = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
        with open(os.path.join(modelRunsDir, 'flows_2011_rail.geojson'), 'w') as f:
            dump(flow_rail, f)

        print("JtW model", Scenario, "cbar [roads, bus, rail] = ", cbar_k)
        print("JtW model", Scenario, "beta [roads, bus, rail] = ", beta_k)

        # Calculate predicted population
        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zonei'] = OXF_MSOA_list

        end = time.perf_counter()
        print("Journey to work model", Scenario, "run elapsed time (secs) =", end - start)
        print()

        return beta_k, DjPred

    elif Scenario == 'NewHousingDev_2019':
        # Use cij as cost matrix (MSOA to MSOA)
        m, n = cij_road_OXF.shape
        model = QUANTJobsModel(m, n)
        model.setAttractorsAj(HHAttractors, 'zonei', 'N_of_Dwellings')
        model.setPopulationEi(jobs, 'zonei', 'employment_tot')
        model.setCostMatrixCij(cij_road_OXF, cij_bus_OXF, cij_rail_OXF)

        Tij, cbar_k = model.run3modes_NoCalibration(Beta_calibrated)
        # Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_road = Tij[0].sum(axis=1)
        Ji_road = Calculate_Job_Accessibility(DjPred_road, cij_road_OXF)

        DjPred_bus = Tij[1].sum(axis=1)
        Ji_bus = Calculate_Job_Accessibility(DjPred_bus, cij_bus_OXF)

        DjPred_rail = Tij[2].sum(axis=1)
        Ji_rail = Calculate_Job_Accessibility(DjPred_rail, cij_rail_OXF)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'areakey': OXF_MSOA_list, 'JAcar19': Ji_road, 'JAbus19': Ji_bus, 'JArail19': Ji_rail})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibility2019"])

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.

        OiPred_road = Tij[0].sum(axis=0)
        Hi_road = Calculate_Housing_Accessibility(OiPred_road, cij_road_OXF)

        OiPred_bus = Tij[1].sum(axis=0)
        Hi_bus = Calculate_Housing_Accessibility(OiPred_bus, cij_bus_OXF)

        OiPred_rail = Tij[2].sum(axis=0)
        Hi_rail = Calculate_Housing_Accessibility(OiPred_rail, cij_bus_OXF)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'areakey': OXF_MSOA_list, 'HAcar19': Hi_road, 'HAbus19': Hi_bus, 'HArail19': Hi_rail})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibility2019"])

        # Create a Oi Dj table
        jobs['DjPred_Car_19'] = Tij[0].sum(axis=1)
        jobs['DjPred_Bus_19'] = Tij[1].sum(axis=1)
        jobs['DjPred_Rail_19'] = Tij[2].sum(axis=1)
        jobs['DjPred_Tot_19'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1) + Tij[2].sum(axis=1)
        jobs['OiPred_Car_19'] = Tij[0].sum(axis=0)
        jobs['OiPred_Bus_19'] = Tij[1].sum(axis=0)
        jobs['OiPred_Rail_19'] = Tij[2].sum(axis=0)
        jobs['OiPred_Tot_19'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis=0)
        jobs['Job_accessibility_roads'] = Jobs_accessibility_df['JAcar19']
        jobs['Jobs_accessibility_bus'] = Jobs_accessibility_df['JAbus19']
        jobs['Jobs_accessibility_rail'] = Jobs_accessibility_df['JArail19']
        jobs['Housing_accessibility_roads'] = Housing_accessibility_df['HAcar19']
        jobs['Housing_accessibility_bus'] = Housing_accessibility_df['HAbus19']
        jobs['Housing_accessibility_rail'] = Housing_accessibility_df['HArail19']
        jobs.to_csv(outputs["JobsDjOi2019"])

        # Save output matrices
        print("Saving output matrices...")

        # Probabilities:
        np.savetxt(outputs["JobsProbTijRoads2019"], jobs_probTij[0], delimiter=",")
        np.savetxt(outputs["JobsProbTijBus2019"], jobs_probTij[1], delimiter=",")
        np.savetxt(outputs["JobsProbTijRail2019"], jobs_probTij[2], delimiter=",")

        # People flows
        np.savetxt(outputs["JobsTijRoads2019"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsTijBus2019"], Tij[1], delimiter=",")
        np.savetxt(outputs["JobsTijRail2019"], Tij[2], delimiter=",")

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and GB grid indexes as
        # ZoneCodes_ATH does not contain the information
        flow_zonecodes = pd.read_csv(inputs["ZonesCoordinates"])
        flow_car = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(outputs["ArrowsFlowsCar2019"], 'w') as f:
            dump(flow_car, f)
        flow_bus = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(outputs["ArrowsFlowsBus2019"], 'w') as f:
            dump(flow_bus, f)
        flow_rail = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
        with open(outputs["ArrowsFlowsRail2019"], 'w') as f:
            dump(flow_rail, f)

        print("JtW model", Scenario, " cbar [roads, bus, rail] = ", cbar_k)

        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zonei'] = OXF_MSOA_list

        end = time.perf_counter()
        print("Journey to work model run elapsed time (secs) =", end - start)
        print()

        return Beta_calibrated, DjPred

    elif Scenario == 'NewHousingDev_2030':
        # Use cij as cost matrix (MSOA to MSOA)
        m, n = cij_road_OXF.shape
        model = QUANTJobsModel(m, n)
        model.setAttractorsAj(HHAttractors, 'zonei', 'N_of_Dwellings')
        model.setPopulationEi(jobs, 'zonei', 'employment_tot')
        model.setCostMatrixCij(cij_road_OXF, cij_bus_OXF, cij_rail_OXF)

        Tij, cbar_k = model.run3modes_NoCalibration(Beta_calibrated)
        # Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_road = Tij[0].sum(axis=1)
        Ji_road = Calculate_Job_Accessibility(DjPred_road, cij_road_OXF)

        DjPred_bus = Tij[1].sum(axis=1)
        Ji_bus = Calculate_Job_Accessibility(DjPred_bus, cij_bus_OXF)

        DjPred_rail = Tij[2].sum(axis=1)
        Ji_rail = Calculate_Job_Accessibility(DjPred_rail, cij_rail_OXF)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'areakey': OXF_MSOA_list, 'JAcar30': Ji_road, 'JAbus30': Ji_bus, 'JArail30': Ji_rail})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibility2030"])

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.

        OiPred_road = Tij[0].sum(axis=0)
        Hi_road = Calculate_Housing_Accessibility(OiPred_road, cij_road_OXF)

        OiPred_bus = Tij[1].sum(axis=0)
        Hi_bus = Calculate_Housing_Accessibility(OiPred_bus, cij_bus_OXF)

        OiPred_rail = Tij[2].sum(axis=0)
        Hi_rail = Calculate_Housing_Accessibility(OiPred_rail, cij_bus_OXF)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'areakey': OXF_MSOA_list, 'HAcar30': Hi_road, 'HAbus30': Hi_bus,'HArail30': Hi_rail})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibility2030"])

        # Create a Oi Dj table
        jobs['DjPred_Car_30'] = Tij[0].sum(axis=1)
        jobs['DjPred_Bus_30'] = Tij[1].sum(axis=1)
        jobs['DjPred_Rail_30'] = Tij[2].sum(axis=1)
        jobs['DjPred_Tot_30'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1) + Tij[2].sum(axis=1)
        jobs['OiPred_Car_30'] = Tij[0].sum(axis=0)
        jobs['OiPred_Bus_30'] = Tij[1].sum(axis=0)
        jobs['OiPred_Rail_30'] = Tij[2].sum(axis=0)
        jobs['OiPred_Tot_30'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis=0)
        jobs['Job_accessibility_roads'] = Jobs_accessibility_df['JAcar30']
        jobs['Jobs_accessibility_bus'] = Jobs_accessibility_df['JAbus30']
        jobs['Jobs_accessibility_rail'] = Jobs_accessibility_df['JArail30']
        jobs['Housing_accessibility_roads'] = Housing_accessibility_df['HAcar30']
        jobs['Housing_accessibility_bus'] = Housing_accessibility_df['HAbus30']
        jobs['Housing_accessibility_rail'] = Housing_accessibility_df['HArail30']
        jobs.to_csv(outputs["JobsDjOi2030"])


        # Save output matrices
        print("Saving output matrices...")

        # Probabilities:
        np.savetxt(outputs["JobsProbTijRoads2030"], jobs_probTij[0], delimiter=",")
        np.savetxt(outputs["JobsProbTijBus2030"], jobs_probTij[1], delimiter=",")
        np.savetxt(outputs["JobsProbTijRail2030"], jobs_probTij[2], delimiter=",")

        # People flows
        np.savetxt(outputs["JobsTijRoads2030"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsTijBus2030"], Tij[1], delimiter=",")
        np.savetxt(outputs["JobsTijRail2030"], Tij[2], delimiter=",")

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and GB grid indexes as
        # ZoneCodes_ATH does not contain the information
        flow_zonecodes = pd.read_csv(inputs["ZonesCoordinates"])
        flow_car = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(outputs["ArrowsFlowsCar2030"], 'w') as f:
            dump(flow_car, f)
        flow_bus = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(outputs["ArrowsFlowsBus2030"], 'w') as f:
            dump(flow_bus, f)
        flow_rail = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
        with open(outputs["ArrowsFlowsRail2030"], 'w') as f:
            dump(flow_rail, f)

        print("JtW model", Scenario, " cbar [roads, bus, rail] = ", cbar_k)

        # Calculate predicted population
        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zonei'] = OXF_MSOA_list

        end = time.perf_counter()
        print("Journey to work model run elapsed time (secs)=", end - start)
        # print("Journey to work model run elapsed time (mins)=", (end - start) / 60)
        print()

        return Beta_calibrated, DjPred

################################################################################
# Retail Model
################################################################################

"""
runPopulationRetailModel
"""
def runPopulationRetailModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_input, inputs, outputs, Scenario_pop_table=None, Scenario='2011'):
    print()
    print("Running Retail-population model.")
    start = time.perf_counter()

    # Population data - from 2011 census
    dfPopMSOAPopulation_EWS = pd.read_csv(data_totalpopulation, usecols=['msoaiz', 'count_allpeople'], index_col='msoaiz')  # EWS
    # Extract Oxfordshire:
    dfPopMSOAPopulation_OXF = dfPopMSOAPopulation_EWS.loc[OXF_MSOA_list]  # Filter rows
    dfPopMSOAPopulation_OXF.sort_index(inplace=True)
    dfPopMSOAPopulation_OXF['msoaiz'] = dfPopMSOAPopulation_OXF.index  # turn the index (i.e. MSOA codes) back into a columm
    dfPopMSOAPopulation_OXF.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
    popretailPopulation = dfPopMSOAPopulation_OXF.join(other=zonecodes_EWS.set_index('areakey'), on='msoaiz')  # join with the zone codes to add zonei col

    if Scenario == '2011':
        popretailPopulation.to_csv(data_populationretail_population_OXF_2011) # columns = [count_allpeople, msoaiz, zonei, lat, lon, osgb36_east, osgb36_north, area]

    elif Scenario == 'NewHousingDev_2019':
        Additional_2019_pop = Scenario_pop_table # columns = ['MSOA', 'NewPop_2011_2019'] if after JtW: ['zonei', 'NewPop_2011_2019']
        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2019_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2019_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True) # Rename Scenario_pop_table's columns

        # Add new population:
        popretailPopulation = dfPopMSOAPopulation_OXF.join(Additional_2019_pop.set_index('msoaiz'), on=['msoaiz'])
        popretailPopulation['NewPop_2011_2019'] = popretailPopulation['NewPop_2011_2019'].fillna(0)  # Replace NaN with 0
        popretailPopulation['count_allpeople'] = popretailPopulation['count_allpeople'] + popretailPopulation['NewPop_2011_2019'] # Sum the additional pop to previous one
        popretailPopulation.rename(columns={'msoaiz': 'zonei'}, inplace=True) # Go back to original name

        # Join and save file:
        popretailPopulation.to_csv(data_populationretail_population_OXF_2019)

    elif Scenario == 'NewHousingDev_2030':
        Additional_2030_pop = Scenario_pop_table  # columns = ['MSOA', 'NewPop_2011_2030'] if after JtW: ['zonei', 'NewPop_2011_2030']
        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2030_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2030_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns

        # Add new population:
        popretailPopulation = dfPopMSOAPopulation_OXF.join(Additional_2030_pop.set_index('msoaiz'), on=['msoaiz'])
        popretailPopulation['NewPop_2011_2030'] = popretailPopulation['NewPop_2011_2030'].fillna(0)  # Replace NaN with 0
        popretailPopulation['count_allpeople'] = popretailPopulation['count_allpeople'] + popretailPopulation['NewPop_2011_2030'] # Sum the additional pop to previous one
        popretailPopulation.rename(columns={'msoaiz': 'zonei'}, inplace=True) # Go back to original name

        # Join and save file:
        popretailPopulation.to_csv(data_populationretail_population_OXF_2030)

    # Extract OXfordshire by postcode from EWS open data:
    if not os.path.isfile(data_open_geolytix_regression_OXF):
        retailpoints_EWS = pd.read_csv(inputs["DataOpenGeolytixRegression"])
        # Import Oxfordshire postcodes as a dataframe:
        OXF_postcodes_df = pd.read_csv(inputs["OxfPostcodes"])  # df containing: Postcode,Latitude,Longitude,Eastings,Northings
        OXF_postcodes_df['DistrictCode'] = OXF_postcodes_df['Postcode'].str[:-3]  # create column with district code
        OXF_DistrictCodes = OXF_postcodes_df['DistrictCode'].tolist()  # save the district codes into a list
        open_geolitix_OXF = retailpoints_EWS.loc[retailpoints_EWS['postcode'].str[:-4].isin(OXF_DistrictCodes)]  # Filter rows (-4 because of the blank space in the middle of the postcode)
        open_geolitix_OXF.reset_index(drop=True, inplace=True)
        open_geolitix_OXF.to_csv(data_open_geolytix_regression_OXF)

    # load the Geolytix retail points file and make an attraction vector from the floorspace
    popretailZones, popretailAttractors = QUANTRetailModel.loadGeolytixData(data_open_geolytix_regression_OXF)  # and this is the open data
    popretailZones.to_csv(data_populationretail_zones) #NOTE: saving largely for data completeness - these two are identical to the retail points model of income
    popretailAttractors.to_csv(data_populationretail_attractors)

    if not (os.path.isfile(data_retailpoints_cij_roads) and os.path.isfile(data_retailpoints_cij_bus) and os.path.isfile(data_retailpoints_cij_rail)):
        retailpoints_cij_roads, retailpoints_cij_bus, retailpoints_cij_rail = costMSOAToPoint_3modes(cij_road_OXF, cij_bus_OXF, cij_rail_OXF, popretailZones, OXF_MSOA_list)  # This takes a while
        saveMatrix(retailpoints_cij_roads, data_retailpoints_cij_roads)
        saveMatrix(retailpoints_cij_bus, data_retailpoints_cij_bus)
        saveMatrix(retailpoints_cij_rail, data_retailpoints_cij_rail)

        np.savetxt(data_retailpoints_cij_roads_csv, retailpoints_cij_roads, delimiter=",")
        np.savetxt(data_retailpoints_cij_bus_csv, retailpoints_cij_bus, delimiter=",")
        np.savetxt(data_retailpoints_cij_rail_csv, retailpoints_cij_rail, delimiter=",")

    else:
        retailpoints_cij_roads = loadMatrix(data_retailpoints_cij_roads)
        retailpoints_cij_bus = loadMatrix(data_retailpoints_cij_bus)
        retailpoints_cij_rail = loadMatrix(data_retailpoints_cij_rail)

    # This is a retail model with poulation instead of income for Ei, then retail zones, attractors and cij are identical

    m, n = retailpoints_cij_roads.shape
    model = QUANTRetailModel(m,n)
    model.setAttractorsAj(popretailAttractors,'zonei','Modelled turnover annual')
    model.setPopulationEi(popretailPopulation,'zonei','count_allpeople')
    model.setCostMatrixCij(retailpoints_cij_roads, retailpoints_cij_bus, retailpoints_cij_rail)
    beta = beta_input # from the Journey to work model calibration

    Rij, cbar = model.run3modes_NoCalibration(beta)

    print("Retail-population model ", Scenario, " cbar [roads, bus, rail] = ", cbar)

    # Compute Probabilities:
    popretail_probRij = model.computeProbabilities3modes(Rij)

    # Save output matrices
    print("Saving output matrices...")

    if Scenario == '2011':
        # Probabilities:
        np.savetxt(data_populationretail_probRij_roads_2011_csv, popretail_probRij[0], delimiter=",")
        np.savetxt(data_populationretail_probRij_bus_2011_csv, popretail_probRij[1], delimiter=",")
        np.savetxt(data_populationretail_probRij_rail_2011_csv, popretail_probRij[2], delimiter=",")

        # Flows
        np.savetxt(data_populationretail_Rij_roads_2011_csv, Rij[0], delimiter=",")
        np.savetxt(data_populationretail_Rij_bus_2011_csv, Rij[1], delimiter=",")
        np.savetxt(data_populationretail_Rij_rail_2011_csv, Rij[2], delimiter=",")


    elif Scenario == 'NewHousingDev_2019':
        # Probabilities:
        np.savetxt(outputs["RetailProbRijRoads2019"], popretail_probRij[0], delimiter=",")
        np.savetxt(outputs["RetailProbRijBus2019"], popretail_probRij[1], delimiter=",")
        np.savetxt(outputs["RetailProbRijRail2019"], popretail_probRij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["RetailRijRoads2019"], Rij[0], delimiter=",")
        np.savetxt(outputs["RetailRijBus2019"], Rij[1], delimiter=",")
        np.savetxt(outputs["RetailRijRail2019"], Rij[2], delimiter=",")

    elif Scenario == 'NewHousingDev_2030':

        # Probabilities:
        np.savetxt(outputs["RetailProbRijRoads2030"], popretail_probRij[0], delimiter=",")
        np.savetxt(outputs["RetailProbRijBus2030"], popretail_probRij[1], delimiter=",")
        np.savetxt(outputs["RetailProbRijRail2030"], popretail_probRij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["RetailRijRoads2030"], Rij[0], delimiter=",")
        np.savetxt(outputs["RetailRijBus2030"], Rij[1], delimiter=",")
        np.savetxt(outputs["RetailRijRail2030"], Rij[2], delimiter=",")

    end = time.perf_counter()
    print("Retail-population model run ", Scenario, " elapsed time (secs) = ", end - start)
    print()

################################################################################
# Schools Model                                                                #
################################################################################

def runSchoolsModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_input, Scenario_pop_table, inputs, outputs, Scenario = '2011'):
    print("runSchoolsModel running primary schools")
    start = time.perf_counter()

    # Population data - from 2011 census
    dfPopMSOAPopulation_EWS = pd.read_csv(data_totalpopulation, usecols=['msoaiz', 'count_allpeople'], index_col='msoaiz')  # EWS
    # Extract Oxfordshire:
    dfPopMSOAPopulation_OXF = dfPopMSOAPopulation_EWS.loc[OXF_MSOA_list]  # Filter rows
    dfPopMSOAPopulation_OXF.sort_index(inplace=True)
    dfPopMSOAPopulation_OXF['msoaiz'] = dfPopMSOAPopulation_OXF.index  # turn the index (i.e. MSOA codes) back into a columm
    dfPopMSOAPopulation_OXF.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
    dfPopMSOAPopulation_OXF = dfPopMSOAPopulation_OXF.join(other=zonecodes_EWS.set_index('areakey'), on='msoaiz')  # join with the zone codes to add zonei col

    # Now schools data:  extract Oxfordshire schools from England data:
    if not os.path.isfile(data_schools_OXF_primary):
        schools_england_primary = pd.read_csv(inputs["DataSchoolsEWSPrimary"], encoding='latin-1')
        # Import Oxfordshire postcodes as a dataframe:
        OXF_postcodes_df = pd.read_csv(inputs["OxfPostcodes"])  # df containing: Postcode,Latitude,Longitude,Eastings,Northings
        OXF_postcodes_df['DistrictCode'] = OXF_postcodes_df['Postcode'].str[:-3]  # create column with district code
        OXF_DistrictCodes = OXF_postcodes_df['DistrictCode'].tolist()  # save the district codes into a list
        schools_OXF_primary = schools_england_primary.loc[schools_england_primary['Postcode'].str[:-4].isin(OXF_DistrictCodes)]  # Filter rows (-4 because of the blank space in the middle of the postcode)
        schools_OXF_primary.reset_index(drop=True, inplace=True)
        schools_OXF_primary.to_csv(data_schools_OXF_primary)

    primaryZones, primaryAttractors = QUANTSchoolsModel.loadSchoolsData(data_schools_OXF_primary)

    row, col = primaryZones.shape
    print("primaryZones count = ", row)

    primaryZones.to_csv(data_primary_zones)
    primaryAttractors.to_csv(data_primary_attractors)

    primaryPopMSOA = pd.read_csv(data_schoolagepopulation, index_col = 'msoaiz') #[,geography code,count_primary,count_secondary] # EWS
    # Extract Oxfordshire:
    primaryPopMSOA = primaryPopMSOA.loc[OXF_MSOA_list]  # Filter rows
    primaryPopMSOA['msoaiz'] = primaryPopMSOA.index  # turn the index (i.e. MSOA codes) back into a columm
    primaryPopMSOA.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!

    primaryPopulation = primaryPopMSOA.join(other=zonecodes_EWS.set_index('areakey'),on='msoaiz')

    if Scenario == '2011':
        primaryPopulation.to_csv(data_primary_population_2011)

    elif Scenario == 'NewHousingDev_2019':
        # Scale population from 2011 counting new houses
        Additional_2019_pop = Scenario_pop_table  # columns = ['MSOA', 'NewPop_2011_2019'] if after JtW: ['zonei', 'NewPop_2011_2019']

        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2019_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2019_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns

        Additional_2019_pop = dfPopMSOAPopulation_OXF.join(Additional_2019_pop.set_index('msoaiz'), on='msoaiz')  # join existing population to additional scenario population

        # Add new population:
        primaryPopulation = primaryPopMSOA.join(Additional_2019_pop.set_index('msoaiz'), on=['msoaiz'])

        primaryPopulation['NewPop_2011_2019'] = primaryPopulation['NewPop_2011_2019'].fillna(0)  # Replace NaN with 0
        primaryPopulation['count_allpeople'] = primaryPopulation['count_allpeople'] + primaryPopulation['NewPop_2011_2019'] # Sum the additional pop to previous one
        primaryPopulation['pop_increment'] = primaryPopulation['NewPop_2011_2019'] / primaryPopulation['count_allpeople'] # Calculate the % population increment
        primaryPopulation['count_primary'] = primaryPopulation['count_primary'] * (1 + primaryPopulation['pop_increment']) # Add the population increment to the number of pupils

        primaryPopulation.to_csv(data_primary_population_2019)

    elif Scenario == 'NewHousingDev_2030':
        # Scale population from 2011 counting new houses
        Additional_2030_pop = Scenario_pop_table  # columns = ['MSOA', 'NewPop_2011_2030'] if after JtW: ['zonei', 'NewPop_2011_2030']

        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2030_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2030_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns

        Additional_2030_pop = dfPopMSOAPopulation_OXF.join(Additional_2030_pop.set_index('msoaiz'), on='msoaiz')  # join existing population to additional scenario population

        # Add new population:
        primaryPopulation = primaryPopMSOA.join(Additional_2030_pop.set_index('msoaiz'), on=['msoaiz'])

        primaryPopulation['NewPop_2011_2030'] = primaryPopulation['NewPop_2011_2030'].fillna(0)  # Replace NaN with 0
        primaryPopulation['count_allpeople'] = primaryPopulation['count_allpeople'] + primaryPopulation['NewPop_2011_2030'] # Sum the additional pop to previous one
        primaryPopulation['pop_increment'] = primaryPopulation['NewPop_2011_2030'] / primaryPopulation['count_allpeople'] # Calculate the % population increment
        primaryPopulation['count_primary'] = primaryPopulation['count_primary'] * (1 + primaryPopulation['pop_increment']) # Add the population increment to the number of pupils

        primaryPopulation.to_csv(data_primary_population_2030)

    if not (os.path.isfile(data_primary_cij_roads) and os.path.isfile(data_primary_cij_bus) and os.path.isfile(data_primary_cij_rail)):
        primary_cij_roads, primary_cij_bus, primary_cij_rail = costMSOAToPoint_3modes(cij_road_OXF, cij_bus_OXF, cij_rail_OXF, primaryZones, OXF_MSOA_list)
        saveMatrix(primary_cij_roads, data_primary_cij_roads)
        saveMatrix(primary_cij_bus, data_primary_cij_bus)
        saveMatrix(primary_cij_rail, data_primary_cij_rail)

        np.savetxt(data_primary_cij_roads_csv, primary_cij_roads, delimiter=",")
        np.savetxt(data_primary_cij_bus_csv, primary_cij_bus, delimiter=",")
        np.savetxt(data_primary_cij_rail_csv, primary_cij_rail, delimiter=",")

    else:
        primary_cij_roads = loadMatrix(data_primary_cij_roads)
        primary_cij_bus = loadMatrix(data_primary_cij_bus)
        primary_cij_rail = loadMatrix(data_primary_cij_rail)

    m, n = primary_cij_roads.shape
    model = QUANTSchoolsModel(m,n)
    model.setAttractorsAj(primaryAttractors,'zonei','SchoolCapacity')
    model.setPopulationEi(primaryPopulation,'zonei','count_primary')
    model.setCostMatrixCij(primary_cij_roads, primary_cij_bus, primary_cij_rail)
    beta = beta_input # from the Journey to work model calibration

    # Pij = pupil flows
    primary_Pij, cbar_primary = model.run3modes_NoCalibration(beta)

    print("Primary schools model ", Scenario, " cbar [roads, bus, rail] = ", cbar_primary)

    # Compute Probabilities:
    primary_probPij = model.computeProbabilities3modes(primary_Pij)

    # Save output matrices
    print("Saving output matrices...")

    if Scenario == '2011':
        # Probabilities:
        np.savetxt(data_primary_probPij_roads_2011_csv, primary_probPij[0], delimiter=",")
        np.savetxt(data_primary_probPij_bus_2011_csv, primary_probPij[1], delimiter=",")
        np.savetxt(data_primary_probPij_rail_2011_csv, primary_probPij[2], delimiter=",")

        # Flows
        np.savetxt(data_primary_Pij_roads_2011_csv, primary_Pij[0], delimiter=",")
        np.savetxt(data_primary_Pij_bus_2011_csv, primary_Pij[1], delimiter=",")
        np.savetxt(data_primary_Pij_rail_2011_csv, primary_Pij[2], delimiter=",")

    elif Scenario == 'NewHousingDev_2019':
        # Probabilities:
        np.savetxt(outputs["PrimaryProbPijRoads2019"], primary_probPij[0], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijBus2019"], primary_probPij[1], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijRail2019"], primary_probPij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["PrimaryPijRoads2019"], primary_Pij[0], delimiter=",")
        np.savetxt(outputs["PrimaryPijBus2019"], primary_Pij[1], delimiter=",")
        np.savetxt(outputs["PrimaryPijRail2019"], primary_Pij[2], delimiter=",")

    elif Scenario == 'NewHousingDev_2030':
        # Probabilities:
        np.savetxt(outputs["PrimaryProbPijRoads2030"], primary_probPij[0], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijBus2030"], primary_probPij[1], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijRail2030"], primary_probPij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["PrimaryPijRoads2030"], primary_Pij[0], delimiter=",")
        np.savetxt(outputs["PrimaryPijBus2030"], primary_Pij[1], delimiter=",")
        np.savetxt(outputs["PrimaryPijRail2030"], primary_Pij[2], delimiter=",")

    end = time.perf_counter()
    print("Primary school model ", Scenario, " run elapsed time (secs) = ", end - start)
    print()

    ###################################################################################################################
    # Secondary schools: it's basically the same code
    print("runSchoolsModel running secondary schools")
    start = time.perf_counter()

    # First extract Oxfordshire schools from England data:
    if not os.path.isfile(data_schools_OXF_secondary):
        schools_england_secondary = pd.read_csv(inputs["DataSchoolsEWSPSecondary"], encoding='latin-1')
        # Import Oxfordshire postcodes as a dataframe:
        OXF_postcodes_df = pd.read_csv(inputs["OxfPostcodes"])  # df containing: Postcode,Latitude,Longitude,Eastings,Northings
        OXF_postcodes_df['DistrictCode'] = OXF_postcodes_df['Postcode'].str[:-3]  # create column with district code
        OXF_DistrictCodes = OXF_postcodes_df['DistrictCode'].tolist()  # save the district codes into a list
        schools_england_secondary = schools_england_secondary.loc[schools_england_secondary['Postcode'].str[:-4].isin(OXF_DistrictCodes)]  # Filter rows (-4 because of the blank space in the middle of the postcode)
        schools_england_secondary.reset_index(drop=True, inplace=True)
        schools_england_secondary.to_csv(data_schools_OXF_secondary)

    secondaryZones, secondaryAttractors = QUANTSchoolsModel.loadSchoolsData(data_schools_OXF_secondary)

    row, col = secondaryZones.shape
    print("secondaryZones count = ", row)
    # print("secondaryZones max = ", secondaryZones.max(axis=0))

    secondaryZones.to_csv(data_secondary_zones)
    secondaryAttractors.to_csv(data_secondary_attractors)

    secondaryPopMSOA = pd.read_csv(data_schoolagepopulation, index_col='msoaiz') # EWS
    # Extract Oxfordshire:
    secondaryPopMSOA = secondaryPopMSOA.loc[OXF_MSOA_list]  # Filter rows
    secondaryPopMSOA['msoaiz'] = secondaryPopMSOA.index  # turn the index (i.e. MSOA codes) back into a columm
    secondaryPopMSOA.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!

    secondaryPopulation = secondaryPopMSOA.join(other=zonecodes_EWS.set_index('areakey'), on='msoaiz')

    if Scenario == '2011':
        secondaryPopulation.to_csv(data_secondary_population_2011)

    elif Scenario == 'NewHousingDev_2019':
        # Scale population from 2011 counting new houses
        Additional_2019_pop = Scenario_pop_table  # columns = ['MSOA', 'NewPop_2011_2019'] if after JtW: ['zonei', 'NewPop_2011_2019']

        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2019_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2019_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns

        Additional_2019_pop = dfPopMSOAPopulation_OXF.join(Additional_2019_pop.set_index('msoaiz'), on='msoaiz')  # join with the zone codes to add zonei col

        # Add new population:
        secondaryPopulation = secondaryPopMSOA.join(Additional_2019_pop.set_index('msoaiz'), on=['msoaiz'])
        secondaryPopulation['NewPop_2011_2019'] = secondaryPopulation['NewPop_2011_2019'].fillna(0)  # Replace NaN with 0
        secondaryPopulation['count_allpeople'] = secondaryPopulation['count_allpeople'] + secondaryPopulation['NewPop_2011_2019']  # Sum the additional pop to previous one
        secondaryPopulation['pop_increment'] = secondaryPopulation['NewPop_2011_2019'] / secondaryPopulation['count_allpeople']  # Calculate the % population increment
        secondaryPopulation['count_secondary'] = secondaryPopulation['count_secondary'] * (1 + secondaryPopulation['pop_increment'])  # Add the population increment to the number of pupils

        secondaryPopulation.to_csv(data_secondary_population_2019)

    elif Scenario == 'NewHousingDev_2030':
        # Scale population from 2011 counting new houses

        Additional_2030_pop = Scenario_pop_table  # columns = ['MSOA', 'NewPop_2011_2030'] if after JtW: ['zonei', 'NewPop_2011_2030']

        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2030_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2030_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns

        Additional_2030_pop = dfPopMSOAPopulation_OXF.join(Additional_2030_pop.set_index('msoaiz'), on='msoaiz')  # join with the zone codes to add zonei col

        # Add new population:
        secondaryPopulation = secondaryPopMSOA.join(Additional_2030_pop.set_index('msoaiz'), on=['msoaiz'])
        secondaryPopulation['NewPop_2011_2030'] = secondaryPopulation['NewPop_2011_2030'].fillna(0)  # Replace NaN with 0
        secondaryPopulation['count_allpeople'] = secondaryPopulation['count_allpeople'] + secondaryPopulation['NewPop_2011_2030']  # Sum the additional pop to previous one
        secondaryPopulation['pop_increment'] = secondaryPopulation['NewPop_2011_2030'] / secondaryPopulation['count_allpeople']  # Calculate the % population increment
        secondaryPopulation['count_secondary'] = secondaryPopulation['count_secondary'] * (1 + secondaryPopulation['pop_increment'])  # Add the population increment to the number of pupils

        secondaryPopulation.to_csv(data_secondary_population_2030)

    if not (os.path.isfile(data_secondary_cij_roads) and os.path.isfile(data_secondary_cij_bus) and os.path.isfile(data_secondary_cij_rail)):
        secondary_cij_roads ,secondary_cij_bus, secondary_cij_rail = costMSOAToPoint_3modes(cij_road_OXF, cij_bus_OXF, cij_rail_OXF, secondaryZones, OXF_MSOA_list)
        saveMatrix(secondary_cij_roads, data_secondary_cij_roads)
        saveMatrix(secondary_cij_bus, data_secondary_cij_bus)
        saveMatrix(secondary_cij_rail, data_secondary_cij_rail)

        np.savetxt(data_secondary_cij_roads_csv, secondary_cij_roads, delimiter=",")
        np.savetxt(data_secondary_cij_bus_csv, secondary_cij_bus, delimiter=",")
        np.savetxt(data_secondary_cij_rail_csv, secondary_cij_rail, delimiter=",")
    else:
        secondary_cij_roads = loadMatrix(data_secondary_cij_roads)
        secondary_cij_bus = loadMatrix(data_secondary_cij_bus)
        secondary_cij_rail = loadMatrix(data_secondary_cij_rail)

    m, n = secondary_cij_roads.shape
    model = QUANTSchoolsModel(m,n)
    model.setAttractorsAj(secondaryAttractors,'zonei','SchoolCapacity')
    model.setPopulationEi(secondaryPopulation,'zonei','count_secondary')
    model.setCostMatrixCij(secondary_cij_roads ,secondary_cij_bus, secondary_cij_rail)
    beta = beta_input # from the Journey to work model calibration

    secondary_Pij, cbar_secondary = model.run3modes_NoCalibration(beta)

    print("Secondary schools model ", Scenario, " cbar [roads, bus, rail] = ", cbar_secondary)

    # Compute Probabilities:
    secondary_probPij = model.computeProbabilities3modes(secondary_Pij)

    # Save output matrices
    print("Saving output matrices...")

    if Scenario == '2011':
        # Probabilities:
        np.savetxt(data_secondary_probPij_roads_2011_csv, secondary_probPij[0], delimiter=",")
        np.savetxt(data_secondary_probPij_bus_2011_csv, secondary_probPij[1], delimiter=",")
        np.savetxt(data_secondary_probPij_rail_2011_csv, secondary_probPij[2], delimiter=",")

        # Flows
        np.savetxt(data_secondary_Pij_roads_2011_csv, secondary_Pij[0], delimiter=",")
        np.savetxt(data_secondary_Pij_bus_2011_csv, secondary_Pij[1], delimiter=",")
        np.savetxt(data_secondary_Pij_rail_2011_csv, secondary_Pij[2], delimiter=",")

    elif Scenario == 'NewHousingDev_2019':
        # Probabilities:
        np.savetxt(outputs["SecondaryProbPijRoads2019"], secondary_probPij[0], delimiter=",")
        np.savetxt(outputs["SecondaryProbPijBus2019"], secondary_probPij[1], delimiter=",")
        np.savetxt(outputs["SecondaryProbPijRail2019"], secondary_probPij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["SecondaryPijRoads2019"], secondary_Pij[0], delimiter=",")
        np.savetxt(outputs["SecondaryPijBus2019"], secondary_Pij[1], delimiter=",")
        np.savetxt(outputs["SecondaryPijRail2019"], secondary_Pij[2], delimiter=",")

    elif Scenario == 'NewHousingDev_2030':
        # Probabilities:
        np.savetxt(outputs["SecondaryProbPijRoads2030"], secondary_probPij[0], delimiter=",")
        np.savetxt(outputs["SecondaryProbPijBus2030"], secondary_probPij[1], delimiter=",")
        np.savetxt(outputs["SecondaryProbPijRail2030"], secondary_probPij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["SecondaryPijRoads2030"], secondary_Pij[0], delimiter=",")
        np.savetxt(outputs["SecondaryPijBus2030"], secondary_Pij[1], delimiter=",")
        np.savetxt(outputs["SecondaryPijRail2030"], secondary_Pij[2], delimiter=",")

    end = time.perf_counter()
    print("secondary school model ", Scenario, " run elapsed time (secs) = ", end - start)
    print()

################################################################################
# Hospitals Model                                                              #
################################################################################

def runHospitalsModel(OXF_MSOA_list, zonecodes_EWS, cij_road_OXF, cij_bus_OXF, cij_rail_OXF, beta_input, Scenario_pop_table, inputs, outputs, Scenario = '2011'):
    print("Running Hospitals model")
    start = time.perf_counter()
    # hospitals model

    # First extract Oxfordshire Hospitals from England data:
    if not os.path.isfile(data_hospitals_OXF):
        hospitals_England = pd.read_csv(inputs["DataHospitals"]) # encoding='latin-1'
        # Import Oxfordshire postcodes as a dataframe:
        OXF_postcodes_df = pd.read_csv(inputs["OxfPostcodes"])  # df containing: Postcode,Latitude,Longitude,Eastings,Northings
        OXF_postcodes_df['DistrictCode'] = OXF_postcodes_df['Postcode'].str[:-3]  # create column with district code
        OXF_DistrictCodes = OXF_postcodes_df['DistrictCode'].tolist()  # save the district codes into a list
        hospitals_OXF = hospitals_England.loc[hospitals_England['pcd'].str[:-4].isin(OXF_DistrictCodes)]  # Filter rows (-4 because of the blank space in the middle of the postcode)
        hospitals_OXF.reset_index(drop=True, inplace=True)
        hospitals_OXF.to_csv(data_hospitals_OXF)

    # load hospitals population
    hospitalZones, hospitalAttractors = QUANTHospitalsModel.loadHospitalsData(data_hospitals_OXF)

    row, col = hospitalZones.shape
    print("hospitalZones count = ",row)

    hospitalZones.to_csv(data_hospital_zones)
    hospitalAttractors.to_csv(data_hospital_attractors)

    hospitalPopMSOA = pd.read_csv(data_totalpopulation, usecols=['msoaiz', 'count_allpeople'], index_col='msoaiz')  # this is the census total count of people
    hospitalPopMSOA = hospitalPopMSOA.loc[OXF_MSOA_list]  # Filter rows: extract Oxfordshire:
    hospitalPopMSOA['msoaiz'] = hospitalPopMSOA.index  # turn the index (i.e. MSOA codes) back into a columm
    hospitalPopMSOA.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
    hospitalPopulation = hospitalPopMSOA.join(other=zonecodes_EWS.set_index('areakey'), on='msoaiz')  # zone with the zone codes to add zonei col

    if Scenario == '2011':
        pass

    elif Scenario == 'NewHousingDev_2019':
        Additional_2019_pop = Scenario_pop_table  # columns = ['MSOA', 'NewPop_2011_2019'] if after JtW: ['zonei', 'NewPop_2011_2019']
        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2019_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2019_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns

        # Add new population:
        hospitalPopulation = hospitalPopMSOA.join(Additional_2019_pop.set_index('msoaiz'), on=['msoaiz'])
        hospitalPopulation['NewPop_2011_2019'] = hospitalPopulation['NewPop_2011_2019'].fillna(0)  # Replace NaN with 0
        hospitalPopulation['count_allpeople'] = hospitalPopulation['count_allpeople'] + hospitalPopulation['NewPop_2011_2019']  # Sum the additional pop to previous one
        hospitalPopulation.rename(columns={'msoaiz': 'zonei'}, inplace=True)  # Go back to original name

    elif Scenario == 'NewHousingDev_2030':
        Additional_2030_pop = Scenario_pop_table  # columns = ['MSOA', 'NewPop_2011_2030'] if after JtW: ['zonei', 'NewPop_2011_2030']
        # Change column name: original name='MSOA', but if thr JtW model has been run already--> column name='zonei'
        Additional_2030_pop.rename(columns={'MSOA': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns
        Additional_2030_pop.rename(columns={'zonei': 'msoaiz'}, inplace=True)  # Rename Scenario_pop_table's columns

        # Add new population:
        hospitalPopulation = hospitalPopMSOA.join(Additional_2030_pop.set_index('msoaiz'), on=['msoaiz'])
        hospitalPopulation['NewPop_2011_2030'] = hospitalPopulation['NewPop_2011_2030'].fillna(0)  # Replace NaN with 0
        hospitalPopulation['count_allpeople'] = hospitalPopulation['count_allpeople'] + hospitalPopulation['NewPop_2011_2030']  # Sum the additional pop to previous one
        hospitalPopulation.rename(columns={'msoaiz': 'zonei'}, inplace=True)  # Go back to original name

    if not (os.path.isfile(data_hospital_cij_roads) and os.path.isfile(data_hospital_cij_bus) and os.path.isfile(data_hospital_cij_rail)):
        hospital_cij_roads, hospital_cij_bus, hospital_cij_rail = costMSOAToPoint_3modes(cij_road_OXF, cij_bus_OXF, cij_rail_OXF, hospitalZones, OXF_MSOA_list)
        saveMatrix(hospital_cij_roads, data_hospital_cij_roads)
        saveMatrix(hospital_cij_bus, data_hospital_cij_bus)
        saveMatrix(hospital_cij_rail, data_hospital_cij_rail)

        np.savetxt(data_hospital_cij_roads_csv, hospital_cij_roads, delimiter=",")
        np.savetxt(data_hospital_cij_bus_csv, hospital_cij_bus, delimiter=",")
        np.savetxt(data_hospital_cij_rail_csv, hospital_cij_rail, delimiter=",")

    else:
        hospital_cij_roads = loadMatrix(data_hospital_cij_roads)
        hospital_cij_bus = loadMatrix(data_hospital_cij_bus)
        hospital_cij_rail = loadMatrix(data_hospital_cij_rail)

    m, n = hospital_cij_roads.shape
    model = QUANTHospitalsModel(m,n)
    model.setAttractorsAj(hospitalAttractors,'zonei','floor_area_m2')
    model.setPopulationEi(hospitalPopulation,'zonei','count_allpeople')
    model.setCostMatrixCij(hospital_cij_roads, hospital_cij_bus, hospital_cij_rail)
    beta = beta_input # from the Journey to work model calibration

    # Hij = hospital flows
    hospital_Hij, cbar = model.run3modes_NoCalibration(beta)

    print("Hospitals model ", Scenario, " cbar [roads, bus, rail] = ", cbar)

    # Compute Probabilities:
    hospital_probHij = model.computeProbabilities3modes(hospital_Hij)

    # Save output matrices
    print("Saving output matrices...")

    if Scenario == '2011':
        # Probabilities:
        np.savetxt(data_hospital_probHij_roads_2011_csv, hospital_probHij[0], delimiter=",")
        np.savetxt(data_hospital_probHij_bus_2011_csv, hospital_probHij[1], delimiter=",")
        np.savetxt(data_hospital_probHij_rail_2011_csv, hospital_probHij[2], delimiter=",")

        # Flows
        np.savetxt(data_hospital_Hij_roads_2011_csv, hospital_Hij[0], delimiter=",")
        np.savetxt(data_hospital_Hij_bus_2011_csv, hospital_Hij[1], delimiter=",")
        np.savetxt(data_hospital_Hij_rail_2011_csv, hospital_Hij[2], delimiter=",")

    elif Scenario == 'NewHousingDev_2019':
        # Probabilities:
        np.savetxt(outputs["HospitalProbHijRoads2019"], hospital_probHij[0], delimiter=",")
        np.savetxt(outputs["HospitalProbHijBus2019"], hospital_probHij[1], delimiter=",")
        np.savetxt(outputs["HospitalProbHijRail2019"], hospital_probHij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["HospitalHijRoads2019"], hospital_Hij[0], delimiter=",")
        np.savetxt(outputs["HospitalHijBus2019"], hospital_Hij[1], delimiter=",")
        np.savetxt(outputs["HospitalHijRail2019"], hospital_Hij[2], delimiter=",")

    elif Scenario == 'NewHousingDev_2030':
        # Probabilities:
        np.savetxt(outputs["HospitalProbHijRoads2030"], hospital_probHij[0], delimiter=",")
        np.savetxt(outputs["HospitalProbHijBus2030"] , hospital_probHij[1], delimiter=",")
        np.savetxt(outputs["HospitalProbHijRail2030"], hospital_probHij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["HospitalHijRoads2030"], hospital_Hij[0], delimiter=",")
        np.savetxt(outputs["HospitalHijBus2030"], hospital_Hij[1], delimiter=",")
        np.savetxt(outputs["HospitalHijRail2030"], hospital_Hij[2], delimiter=",")

    end = time.perf_counter()
    print("hospitals model run ", Scenario, " elapsed time (secs) = ", end - start)
    print()


def Calculate_Job_Accessibility(DjPred, cij):
    # Job accessibility is the distribution of population around a job location.
    # It’s just the sum of all the population around a job zone divided by the travel time squared.
    # This is scaled so that the total of all i zones comes to 100.

    Ji = np.zeros(len(DjPred))
    for i in range(len(Ji)):
        for j in range(len(Ji)):
            Ji[i] += DjPred[j] / (cij[i, j] * cij[i, j])  # DjPred is residential totals

    # now scale to 100
    Sum = 0
    for i in range(len(Ji)): Sum += Ji[i]
    for i in range(len(Ji)): Ji[i] = 100.0 * Ji[i] / Sum
    return Ji

def Calculate_Housing_Accessibility(OiPred, cij):
    # Housing accessibility is the distribution of jobs around a housing location.
    # It’s just the sum of all the jobs around a zone divided by the travel time squared.
    # This is scaled so that the total of all i zones comes to 100.

    Hi = np.zeros(len(OiPred))

    # Calculate housing accessibility for public transport
    for i in range(len(Hi)):
        for j in range(len(Hi)):
            Hi[i] += OiPred[j] / (cij[i, j] * cij[i, j])  # OiPred_pu is employment totals

    # now scale to 100
    Sum = 0
    for i in range(len(Hi)): Sum += Hi[i]
    for i in range(len(Hi)): Hi[i] = 100.0 * Hi[i] / Sum
    return Hi

################################################################################
# END OF MAIN PROGRAM                                                          #
################################################################################


