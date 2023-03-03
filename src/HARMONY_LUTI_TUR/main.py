"""
HARMONY Land-Use Transport-Interaction Model - Turin case study
main.py

November 2021
Author: Fulvio D. Lopane, Centre for Advanced Spatial Analysis, University College London
https://www.casa.ucl.ac.uk

- Developed from Richard Milton's QUANT_RAMP
- Further developed from Eleni Kalantzi, Research Assistant, Centre for Advanced Spatial Analysis, University College London
"""
import os
import time
import pandas as pd
from geojson import dump
import numpy as np
from HARMONY_LUTI_TUR.globals import *
from HARMONY_LUTI_TUR.analytics import graphProbabilities, flowArrowsGeoJSON
from HARMONY_LUTI_TUR.quantschoolsmodel import QUANTSchoolsModel
from HARMONY_LUTI_TUR.quanthospitalsmodel import QUANTHospitalsModel
from HARMONY_LUTI_TUR.quantlhmodel import QUANTLHModel
from HARMONY_LUTI_TUR.maps import *


def start_main(inputs, outputs):
    # Initialisation Phase

    # NOTE: this section provides the base data for the models that come later. This
    # will only be run on the first run of the program to assemble all the tables
    # required from the original sources. After that, if the file exists in the
    # directory, then nothing new is created and this section is effectively
    # skipped, up until the model run section at the end.

    # make a model-runs dir if we need it
    if not os.path.exists(modelRunsDir):
        os.makedirs(modelRunsDir)

    # Zone Codes for Turin
    print()
    print("Importing TUR zone codes")

    zonecodes_TUR = pd.read_csv(inputs["ZoneCodesFile"])
    zonecodes_TUR.set_index('ZONE')
    zonecodes_TUR_list = zonecodes_TUR['ZONE'].tolist()

    #_____________________________________________________________________________________
    # IMPORT cij matrices
    # Fisrt, import intrazone distances
    intrazone_dist_df = pd.read_csv(inputs["IntrazoneDist"], usecols=["Intra_dist"])  # create df from csv
    intrazone_dist_list = intrazone_dist_df.Intra_dist.values.tolist()  # save the intrazone distances into a list

    # CAR cij
    print()
    print("Importing car cost matrix (cij) for Turin")

    # Import the csv cij car as Pandas DataFrame:
    cij_car_TUR = pd.read_csv(inputs["CijCarODZones2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_2019 = cij_car_TUR.to_numpy()
    cij_car_2019[cij_car_2019 < 0] = 210  # upper limit of 210 mins (140 max value) - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_car_2019[cij_car_2019 < 1] = 1  # lower limit of 1 minute links
    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_car = 22  # define average speed in km/h - based on Turin's vehicles
    average_TT_car = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_car.append((i / 1000) / (av_speed_car/ 60))  # result in minutes
    np.fill_diagonal(cij_car_2019, average_TT_car)  # save the average TT list as the main diagonal of the cij matrix

    # Print the dimensions of the matrix to check that everything is ok:
    print('cij car shape: ', cij_car_2019.shape)
    cij_car_2019[cij_car_2019 < 1] = 1  # repeat of above after leading diagonal zeros added

    # Primary Schools
    cij_car_TUR_primary = pd.read_csv(inputs["CijCarPrimary2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_primary_2019 = cij_car_TUR_primary.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij car (primary schools) shape: ', cij_car_primary_2019.shape)

    # Middle Schools
    cij_car_TUR_middle = pd.read_csv(inputs["CijCarMiddle2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_middle_2019 = cij_car_TUR_middle.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij car (middle schools) shape: ', cij_car_middle_2019.shape)

    # High Schools
    cij_car_TUR_high = pd.read_csv(inputs["CijCarHigh2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_high_2019 = cij_car_TUR_high.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij car (high schools) shape: ', cij_car_high_2019.shape)

    # Universities
    cij_car_TUR_uni = pd.read_csv(inputs["CijCarUni2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_uni_2019 = cij_car_TUR_uni.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij car (uni) shape: ', cij_car_uni_2019.shape)

    #Hospitals
    # Import the csv cij Car as Pandas DataFrame:
    cij_car_TUR_hosp = pd.read_csv(inputs["CijCarHospitals2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_hosp_2019 = cij_car_TUR_hosp.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij car (hospitals) shape: ', cij_car_hosp_2019.shape)

    #_____________________________________________________________________________________
    # BUS cij
    print()
    print("Importing bus cost matrix (cij) for Turin")

    # Import the csv cij private as Pandas DataFrame:
    cij_bus_TUR = pd.read_csv(inputs["CijBusODZones2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_2019 = cij_bus_TUR.to_numpy()
    cij_bus_2019[cij_bus_2019 < 0] = 285  # upper limit of 285 mins (max value 190 mins) - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_bus_2019[cij_bus_2019 < 1] = 1  # lower limit of 1 minute links
    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_bus = 11  # define average speed in km/h - based on Turin's buses
    average_TT_bus = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_bus.append((i / 1000) / (av_speed_bus/ 60))  # result in minutes
    np.fill_diagonal(cij_bus_2019, average_TT_bus)  # save the average TT list as the main diagonal of the cij matrix
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij bus shape: ', cij_bus_2019.shape)
    cij_bus_2019[cij_bus_2019 < 1] = 1  # repeat of above after leading diagonal zeros added


    # Primary Schools
    cij_bus_TUR_primary = pd.read_csv(inputs["CijBusPrimary2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_primary_2019 = cij_bus_TUR_primary.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij bus (primary schools) shape: ', cij_bus_primary_2019.shape)

    # Middle Schools
    cij_bus_TUR_middle = pd.read_csv(inputs["CijBusMiddle2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_middle_2019 = cij_bus_TUR_middle.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij Bus (middle schools) shape: ', cij_bus_middle_2019.shape)

    # High Schools
    cij_bus_TUR_high = pd.read_csv(inputs["CijBusHigh2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_high_2019 = cij_bus_TUR_high.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij bus (high schools) shape: ', cij_bus_high_2019.shape)

    # Universities
    cij_bus_TUR_uni = pd.read_csv(inputs["CijBusUni2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_uni_2019 = cij_bus_TUR_uni.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij bus (uni) shape: ', cij_bus_uni_2019.shape)

    # Hospitals
    # Import the csv cij bus as Pandas DataFrame:
    cij_bus_TUR_hosp = pd.read_csv(inputs["CijBusHospitals2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_hosp_2019 = cij_bus_TUR_hosp.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij bus (hospitals) shape: ', cij_bus_hosp_2019.shape)

    #_____________________________________________________________________________________
    # RAILWAYS cij
    print()
    print("Importing rail cost matrix (cij) for Turin")

    # Import the csv cij private as Pandas DataFrame:
    cij_rail_TUR = pd.read_csv(inputs["CijRailODZones2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_2019 = cij_rail_TUR.to_numpy()
    cij_rail_2019[cij_rail_2019 < 0] = 342  # upper limit of 342 mins (max value 228 mins) - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_rail_2019[cij_rail_2019 < 1] = 1  # lower limit of 1 minute links
    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_rail = 18  # define average speed in km/h - based on Turin's metro/tram
    average_TT_rail = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_rail.append((i / 1000) / (av_speed_rail/ 60))  # result in minutes
    np.fill_diagonal(cij_rail_2019, average_TT_rail)  # save the average TT list as the main diagonal of the cij matrix
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij rail shape: ', cij_rail_2019.shape)
    cij_rail_2019[cij_rail_2019 < 1] = 1  # repeat of above after leading diagonal zeros added

    # Primary Schools
    cij_rail_TUR_primary = pd.read_csv(inputs["CijRailPrimary2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_primary_2019 = cij_rail_TUR_primary.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij rail (primary schools) shape: ', cij_rail_primary_2019.shape)

    # Middle Schools
    cij_rail_TUR_middle = pd.read_csv(inputs["CijRailMiddle2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_middle_2019 = cij_rail_TUR_middle.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij rail (middle schools) shape: ', cij_rail_middle_2019.shape)

    # High Schools
    cij_rail_TUR_high = pd.read_csv(inputs["CijRailHigh2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_high_2019 = cij_rail_TUR_high.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij rail (high schools) shape: ', cij_rail_high_2019.shape)

    # Universities
    cij_rail_TUR_uni = pd.read_csv(inputs["CijRailUni2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_uni_2019 = cij_rail_TUR_uni.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij rail (uni) shape: ', cij_rail_uni_2019.shape)

    # Hospitals
    # Import the csv cij rail as Pandas DataFrame:
    cij_rail_TUR_hosp = pd.read_csv(inputs["CijRailHospitals2019"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_hosp_2019 = cij_rail_TUR_hosp.to_numpy()
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij rail (hospitals) shape: ', cij_rail_hosp_2019.shape)

    print("Importing cij matrices completed.")
    print()
    #_____________________________________________________________________________________

    # IMPORT SObs matrices: observed trips
    # In the Turin case study instead of observed trips, we have modelled trips:
    # the calibration is against Turin Visum model predicted trips
    print("Importing SObs matrices")

    # SObs CAR
    print()
    print("Importing SObs for Cars for Turin")

    # Import the csv SObs for Cars as Pandas DataFrame:
    Sobs_car_df = pd.read_csv(inputs["ObsCarCommuting"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    SObs_car = Sobs_car_df.to_numpy()
    print('SObs car shape: ', SObs_car.shape)
    #_____________________________________________________________________________________

    # SObs BUS
    print()
    print("Importing SObs for bus for Turin")

    # Import the csv SObs for bus as Pandas DataFrame:
    Sobs_bus_df = pd.read_csv(inputs["ObsBusCommuting"] , header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    SObs_bus = Sobs_bus_df.to_numpy()
    print('SObs bus shape: ', SObs_bus.shape)
    #_____________________________________________________________________________________

    # SObs RAIL
    print()
    print("Importing SObs for rail for Turin")

    # Import the csv SObs for rail as Pandas DataFrame:
    Sobs_rail_df = pd.read_csv(inputs["ObsRailCommuting"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    SObs_rail = Sobs_rail_df.to_numpy()
    print('SObs rail shape: ', SObs_rail.shape)
    #_____________________________________________________________________________________

    # IMPORT 2030 Cij as well HERE

    #Work
    # Import the csv cij car as Pandas DataFrame:
    cij_car_TUR_2030 = pd.read_csv(inputs["CijCarODZones2030"] , header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_2030 = cij_car_TUR_2030.to_numpy()
    cij_car_2030[cij_car_2030 < 0] = 210  # upper limit of 210 mins (140 max value) - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_car_2030[cij_car_2030 < 1] = 1  # lower limit of 1 minute links
    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_car = 22  # define average speed in km/h - based on Turin's vehicles
    average_TT_car = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_car.append((i / 1000) / (av_speed_car / 60))  # result in minutes
    np.fill_diagonal(cij_car_2030, average_TT_car)  # save the average TT list as the main diagonal of the cij matrix

    # Print the dimensions of the matrix to check that everything is ok:
    print('cij car 2030 shape: ', cij_car_2030.shape)
    cij_car_2030[cij_car_2030 < 1] = 1  # repeat of above after leading diagonal zeros added

    # Import the csv cij bus as Pandas DataFrame:
    cij_bus_TUR_2030 = pd.read_csv(inputs["CijBusODZones2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_2030 = cij_bus_TUR_2030.to_numpy()
    cij_bus_2030[cij_bus_2030 < 0] = 285  # upper limit of 285 mins (max value 190 mins) - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_bus_2030[cij_bus_2030 < 1] = 1  # lower limit of 1 minute links
    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_bus = 11  # define average speed in km/h - based on Turin's buses
    average_TT_bus = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_bus.append((i / 1000) / (av_speed_bus/ 60))  # result in minutes
    np.fill_diagonal(cij_bus_2030, average_TT_bus)  # save the average TT list as the main diagonal of the cij matrix
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij bus 2030 shape: ', cij_bus_2030.shape)
    cij_bus_2030[cij_bus_2030 < 1] = 1  # repeat of above after leading diagonal zeros added

    # Import the csv cij rail as Pandas DataFrame:
    cij_rail_TUR_2030 = pd.read_csv(inputs["CijRailODZones2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_2030 = cij_rail_TUR_2030.to_numpy()
    cij_rail_2030[cij_rail_2030 < 0] = 380  # upper limit of 380 mins (max value 255 mins) - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_rail_2030[cij_rail_2030 < 1] = 1  # lower limit of 1 minute links
    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_rail = 18  # define average speed in km/h - based on Turin's metro/tram
    average_TT_rail = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_rail.append((i / 1000) / (av_speed_rail / 60))  # result in minutes
    np.fill_diagonal(cij_rail_2030, average_TT_rail)  # save the average TT list as the main diagonal of the cij matrix
    # Print the dimensions of the matrix to check that everything is ok:
    print('cij rail 2030 shape: ', cij_rail_2030.shape)
    cij_rail_2030[cij_rail_2030 < 1] = 1  # repeat of above after leading diagonal zeros added

    #Schools
    # Import the csv cij Car as Pandas DataFrame:
    cij_car_primary_TUR_2030 = pd.read_csv(inputs["CijCarPrimary2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_primary_2030 = cij_car_primary_TUR_2030.to_numpy()
    print('cij car for primary 2030 shape: ', cij_car_primary_2030.shape)

    # Import the csv cij bus as Pandas DataFrame:
    cij_bus_primary_TUR_2030 = pd.read_csv(inputs["CijBusPrimary2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_primary_2030 = cij_bus_primary_TUR_2030.to_numpy()
    print('cij bus for primary 2030 shape: ', cij_bus_primary_2030.shape)

    # Import the csv cij rail as Pandas DataFrame:
    cij_rail_primary_TUR_2030 = pd.read_csv(inputs["CijRailPrimary2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_primary_2030 = cij_rail_primary_TUR_2030.to_numpy()
    print('cij rail for primary 2030 shape: ', cij_rail_primary_2030.shape)

    # Import the csv cij Car as Pandas DataFrame:
    cij_car_Middle_TUR_2030 = pd.read_csv(inputs["CijCarMiddle2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_Middle_2030 = cij_car_Middle_TUR_2030.to_numpy()
    print('cij car for middle 2030 shape: ', cij_car_Middle_2030.shape)

    # Import the csv cij bus as Pandas DataFrame:
    cij_bus_Middle_TUR_2030 = pd.read_csv(inputs["CijBusMiddle2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_Middle_2030 = cij_bus_Middle_TUR_2030.to_numpy()
    print('cij bus for middle 2030 shape: ', cij_bus_Middle_2030.shape)

    # Import the csv cij rail as Pandas DataFrame:
    cij_rail_Middle_TUR_2030 = pd.read_csv(inputs["CijRailMiddle2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_Middle_2030 = cij_rail_Middle_TUR_2030.to_numpy()
    print('cij rail for middle 2030 shape: ', cij_rail_Middle_2030.shape)

    # Import the csv cij car as Pandas DataFrame:
    cij_car_High_TUR_2030 = pd.read_csv(inputs["CijCarHigh2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_high_2030 = cij_car_High_TUR_2030.to_numpy()
    print('cij car for high 2030 shape: ', cij_car_high_2030.shape)

    # Import the csv cij bus as Pandas DataFrame:
    cij_bus_High_TUR_2030 = pd.read_csv(inputs["CijBusHigh2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_high_2030 = cij_bus_High_TUR_2030.to_numpy()
    print('cij bus for high 2030 shape: ', cij_bus_high_2030.shape)

    # Import the csv cij rail as Pandas DataFrame:
    cij_rail_High_TUR_2030 = pd.read_csv(inputs["CijRailHigh2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_high_2030 = cij_rail_High_TUR_2030.to_numpy()
    print('cij rail for high 2030 shape: ', cij_rail_high_2030.shape)

    # Import the csv cij Car as Pandas DataFrame:
    cij_car_uni_TUR_2030 = pd.read_csv(inputs["CijCarUni2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_uni_2030 = cij_car_uni_TUR_2030.to_numpy()
    print('cij car for unis 2030 shape: ', cij_car_uni_2030.shape)

    # Import the csv cij bus as Pandas DataFrame:
    cij_bus_uni_TUR_2030 = pd.read_csv(inputs["CijBusUni2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_uni_2030 = cij_bus_uni_TUR_2030.to_numpy()
    print('cij bus for unis 2030 shape: ', cij_bus_uni_2030.shape)

    # Import the csv cij rail as Pandas DataFrame:
    cij_rail_uni_TUR_2030 = pd.read_csv(inputs["CijRailUni2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_uni_2030 = cij_rail_uni_TUR_2030.to_numpy()
    print('cij rail for unis 2030 shape: ', cij_rail_uni_2030.shape)

    #Hospitals
    # Import the csv cij Car as Pandas DataFrame:
    cij_car_Hosp_2030 = pd.read_csv(inputs["CijCarHospitals2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_car_hospitals_2030 = cij_car_Hosp_2030.to_numpy()
    print('cij car for hospitals 2030 shape: ', cij_car_hospitals_2030.shape)

    # Import the csv cij bus as Pandas DataFrame:
    cij_bus_Hosp_2030 = pd.read_csv(inputs["CijBusHospitals2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_bus_hospitals_2030 = cij_bus_Hosp_2030.to_numpy()
    print('cij bus for hospitals 2030 shape: ', cij_bus_hospitals_2030.shape)

    # Import the csv cij rail as Pandas DataFrame:
    cij_rail_Hosp_2030 = pd.read_csv(inputs["CijRailHospitals2030"], header=0, index_col=0)
    # Convert the dataframe to a nupmy matrix:
    cij_rail_hospitals_2030 = cij_rail_Hosp_2030.to_numpy()
    print('cij rail for hospitals 2030 shape: ', cij_rail_hospitals_2030.shape)

    # now run the relevant models to produce the outputs
    runNewLandUseandInfrastructure(zonecodes_TUR_list,
                                   SObs_car, SObs_bus, SObs_rail,
                                   cij_car_2019, cij_bus_2019, cij_rail_2019,
                                   cij_car_2030, cij_bus_2030, cij_rail_2030,
                                   cij_car_primary_2019, cij_bus_primary_2019, cij_rail_primary_2019,
                                   cij_car_middle_2019, cij_bus_middle_2019, cij_rail_middle_2019,
                                   cij_car_high_2019, cij_bus_high_2019, cij_rail_high_2019,
                                   cij_car_uni_2019, cij_bus_uni_2019, cij_rail_uni_2019,
                                   cij_car_hosp_2019, cij_bus_hosp_2019, cij_rail_hosp_2019,
                                   cij_car_primary_2030, cij_bus_primary_2030,cij_rail_primary_2030,
                                   cij_car_Middle_2030, cij_bus_Middle_2030, cij_rail_Middle_2030,
                                   cij_car_high_2030, cij_bus_high_2030, cij_rail_high_2030,
                                   cij_car_uni_2030, cij_bus_uni_2030, cij_rail_uni_2030,
                                   cij_car_hospitals_2030, cij_bus_hospitals_2030, cij_rail_hospitals_2030,
                                   inputs, outputs)

    # Create maps outputs:
    population_map_creation(inputs, outputs)

    # Flows maps:
    # THIS FEATURE IS TURNED OFF - very long run time (>3h) - only for HQ flows visualisation
    create_flow_maps = False
    if create_flow_maps:
        flows_output_keys = ["JobsTijRoads2019", "JobsTijBus2019", "JobsTijRoads2030", "JobsTijBus2030"]
        flows_map_creation(inputs, outputs, flows_output_keys)

################################################################################
# New infrastructure and land use change scenario
################################################################################
"""
Scenario
first run the Journey to Work model (for calibration)
then change the population reading from the new housing development table
and make predictions based on the new population 
"""

def runNewLandUseandInfrastructure(zonecodes_TUR_list, SObs_car, SObs_bus, SObs_rail,
                                   cij_car_2019, cij_bus_2019, cij_rail_2019,
                                   cij_car_2030, cij_bus_2030, cij_rail_2030,
                                   cij_car_primary_2019, cij_bus_primary_2019, cij_rail_primary_2019,
                                   cij_car_middle_2019, cij_bus_middle_2019, cij_rail_middle_2019,
                                   cij_car_high_2019, cij_bus_high_2019, cij_rail_high_2019,
                                   cij_car_uni_2019, cij_bus_uni_2019, cij_rail_uni_2019,
                                   cij_car_hosp_2019, cij_bus_hosp_2019, cij_rail_hosp_2019,
                                   cij_car_primary_2030, cij_bus_primary_2030,cij_rail_primary_2030,
                                   cij_car_Middle_2030, cij_bus_Middle_2030, cij_rail_Middle_2030,
                                   cij_car_high_2030, cij_bus_high_2030, cij_rail_high_2030,
                                   cij_car_uni_2030, cij_bus_uni_2030, cij_rail_uni_2030,
                                   cij_car_hospitals_2030, cij_bus_hospitals_2030, cij_rail_hospitals_2030, inputs, outputs):
    # First run the base model to calibrate it with 2019 observed trip data:

    # Run Journey to work model:
    beta_2019, DjPred_JtW_2019 = runJourneyToWorkModel(zonecodes_TUR_list, cij_car_2019, cij_bus_2019, cij_rail_2019, SObs_car, SObs_bus, SObs_rail, inputs, outputs)

    # Run Schools model:
    runPrimarySchoolsModel(cij_car_primary_2019, cij_bus_primary_2019, cij_rail_primary_2019, beta_2019, inputs, outputs, '2019')
    runMiddleSchoolsModel(cij_car_middle_2019, cij_bus_middle_2019, cij_rail_middle_2019, beta_2019, inputs, outputs, '2019')
    runHighSchoolsModel(cij_car_high_2019, cij_bus_high_2019, cij_rail_high_2019, beta_2019, inputs, outputs, '2019')
    runUniversitiesModel(cij_car_uni_2019, cij_bus_uni_2019, cij_rail_uni_2019, beta_2019, inputs, outputs, '2019')

    # Run Hospitals model:
    runHospitalsModel(cij_car_hosp_2019, cij_bus_hosp_2019, cij_rail_hosp_2019, beta_2019, inputs, outputs, '2019')

    # HARMONY new Land Use and Infrastructure development scenario:
    # base year: 2019, projection year: 2030
    # Firstly, read the new employment files for 2030: updated new number of jobs per zone
    # 2030 (proj year): use projection 2030 employment data
    # Secondly, read travel times for 2030: updated travel costs matrices with new infrastructure
    # 2030 (proj year): use projection 2030 SKIM matrices
    # Thirdly, read the hospitals' files for 2030: updated capacity of hospital in the new development zone.
    # 2030 (proj year): use projection 2030 hospitals' data
    # Fourthly, read education files for 2030: the updated universities list (with capacity).
    # 2030 (proj year): use projection 2030 universities' data
    # Then, run the models with this new parameters

    # read csv for new hospitals
    hospitalPopulation_30 = pd.read_csv(inputs["DataPopulation2030"], usecols=['ZONE', 'Pop_2030'], index_col='ZONE')

    hospitalAttractors_30 = pd.read_csv(inputs["DataHospitals2030"], usecols=['zone', 'beds'], index_col='zone')
    # Need to substitute 0 values in capacity dataframe with very low values (e.g. 1) to avoid division by zero:
    hospitalAttractors_30.replace(0, 1, inplace=True)


    # Now run the JtW model with 2019 beta and New Land Use & Infrastructure Development parameters (without calibration this time)

    beta_2019, DjPred_JtW_2030 = runJourneyToWorkModel(zonecodes_TUR_list, cij_car_2030, cij_bus_2030, cij_rail_2030, SObs_car, SObs_bus, SObs_rail, inputs, outputs, 'NewLandUse&Infr2030', beta_2019)


    # NewHousingDev_2030
    # Run Schools model:
    runPrimarySchoolsModel(cij_car_primary_2030, cij_bus_primary_2030, cij_rail_primary_2030, beta_2019, inputs, outputs, 'NewLandUse&Infr2030')
    runMiddleSchoolsModel(cij_car_Middle_2030, cij_bus_Middle_2030, cij_rail_Middle_2030, beta_2019, inputs, outputs, 'NewLandUse&Infr2030')
    runHighSchoolsModel(cij_car_high_2030, cij_bus_high_2030, cij_rail_high_2030, beta_2019, inputs, outputs, 'NewLandUse&Infr2030')
    runUniversitiesModel(cij_car_uni_2030, cij_bus_uni_2030, cij_rail_uni_2030, beta_2019, inputs, outputs,'NewLandUse&Infr2030')

    # Run Hospitals model:
    runHospitalsModel(cij_car_hospitals_2030, cij_bus_hospitals_2030, cij_rail_hospitals_2030, beta_2019, inputs, outputs, 'NewLandUse&Infr2030')


# What follows from here are the different model run functions for journey to work, schools and hospitals.

################################################################################
# Journey to work Model                                                        #
################################################################################

"""
runJourneyToWorkModel
Origins: workplaces, Destinations: households' population
"""
# Journey to work model with households (HH) floorspace as attractor
def runJourneyToWorkModel(zonecodes_TUR_list, cij_car, cij_bus, cij_rail, SObs_car, SObs_bus, SObs_rail, inputs, outputs, Scenario='2019', Beta_calibrated=None):
    print("Running Journey to Work", Scenario, " model.")
    start = time.perf_counter()
    # Singly constrained model:
    # We conserve the number of jobs and predict the population residing in FUA zones
    # journeys to work generated by jobs
    # Origins: workplaces
    # Destinations: FUA households
    # Attractor: residential floorspace

    """
                    Journey to work
     Origins:       workplaces
     Destinations:  households
     conserved:     jobs
     predicted:     population per zone
     attractor:     N of dwellings
    """

    # Now run the model with or without calibration according to the scenario:
    if Scenario == '2019':
        # load jobs data for residential zones
        dfEi = pd.read_csv(inputs["DataEmployment2019"], usecols=['zone', 'employed_2019'], index_col='zone')

        df_floorspace = pd.read_csv(inputs["HhFloorspace2019"], usecols=['Zone', 'Residential_FloorSpace'], index_col='Zone')
        # Need to substitute 0 values in floorspace dataframe with very low values (e.g. 1) to avoid division by zero:
        df_floorspace.replace(0, 1, inplace=True)

        # Use cij as cost matrix
        m, n = cij_car.shape
        model = QUANTLHModel(m, n)
        model.setObsMatrix(SObs_car, SObs_bus, SObs_rail)
        model.setAttractorsAj(df_floorspace, 'Zone', 'Residential_FloorSpace')
        model.setPopulationEi(dfEi, 'zone', 'employed_2019')
        model.setCostMatrixCij(cij_car, cij_bus, cij_rail)

        # Run model
        Tij, beta_k, cbar_k = model.run3modes()

        # Compute the probability of a flow from a zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_Car = Tij[0].sum(axis=1)
        Ji_Car = Calculate_Job_Accessibility(DjPred_Car, cij_car)

        DjPred_bus = Tij[1].sum(axis=1)
        Ji_bus = Calculate_Job_Accessibility(DjPred_bus, cij_bus)

        DjPred_rail = Tij[2].sum(axis=1)
        Ji_rail = Calculate_Job_Accessibility(DjPred_rail, cij_rail)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'zone': zonecodes_TUR_list, 'JAcar19': Ji_Car, 'JAbus19': Ji_bus, 'JArail19': Ji_rail})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibility2019"])

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.

        OiPred_Car = Tij[0].sum(axis=0)
        Hi_Car = Calculate_Housing_Accessibility(OiPred_Car, cij_car)

        OiPred_bus = Tij[1].sum(axis=0)
        Hi_bus = Calculate_Housing_Accessibility(OiPred_bus, cij_bus)

        OiPred_rail = Tij[2].sum(axis=0)
        Hi_rail = Calculate_Housing_Accessibility(OiPred_rail, cij_rail)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_TUR_list, 'HAcar19': Hi_Car, 'HAbus19': Hi_bus, 'HArail19': Hi_rail})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibility2019"])

        # Create a Oi Dj table
        dfEi['DjPred_Cars_19'] = Tij[0].sum(axis=1)
        dfEi['DjPred_Bus_19'] = Tij[1].sum(axis=1)
        dfEi['DjPred_Rail_19'] = Tij[2].sum(axis=1)
        dfEi['DjPred_Tot_19'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1) + Tij[2].sum(axis=1)
        dfEi['OiPred_Cars_19'] = Tij[0].sum(axis=0)
        dfEi['OiPred_Bus_19'] = Tij[1].sum(axis=0)
        dfEi['OiPred_Rail_19'] = Tij[2].sum(axis=0)
        dfEi['OiPred_Tot_19'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis=0)
        dfEi['Job_accessibility_Cars'] = Jobs_accessibility_df['JAcar19']
        dfEi['Jobs_accessibility_bus'] = Jobs_accessibility_df['JAbus19']
        dfEi['Jobs_accessibility_rail'] = Jobs_accessibility_df['JArail19']
        dfEi['Housing_accessibility_Cars'] = Housing_accessibility_df['HAcar19']
        dfEi['Housing_accessibility_bus'] = Housing_accessibility_df['HAbus19']
        dfEi['Housing_accessibility_rail'] = Housing_accessibility_df['HArail19']
        dfEi.to_csv(outputs["JobsDjOi2019"])

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
        # I need my own zone codes file containing the zonei and greek grid indexes as
        # ZoneCodes_ATH does not contain the information
        flow_zonecodes = pd.read_csv(inputs["ZonesCoordinates"])
        flow_car = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(outputs["ArrowsFlowsRoads2019"], 'w') as f:
            dump(flow_car, f)
        flow_bus = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(outputs["ArrowsFlowsBus2019"], 'w') as f:
            dump(flow_bus, f)
        flow_rail = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
        with open(outputs["ArrowsFlowsRail2019"], 'w') as f:
            dump(flow_rail, f)

        print("JtW model", Scenario, "cbar [Cars, bus, rail] = ", cbar_k)
        print("JtW model", Scenario, "beta [Cars, bus, rail] = ", beta_k)

        # Calculate predicted population
        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_TUR_list

        end = time.perf_counter()
        print("Journey to work model", Scenario, "run elapsed time (secs) =", end - start)
        print()

        return beta_k, DjPred

    elif Scenario == 'NewLandUse&Infr2030':
        # load jobs data for residential zones
        dfEi = pd.read_csv(inputs["DataEmployment2030"], usecols=['zone', 'employed'], index_col='zone')

        df_floorspace = pd.read_csv(inputs["HhFloorspace2019"], usecols=['Zone', 'Residential_FloorSpace'], index_col='Zone')
        # Need to substitute 0 values in floorspace dataframe with very low values (e.g. 1) to avoid division by zero:
        df_floorspace.replace(0, 1, inplace=True)

        # Use cij as cost matrix
        m, n = cij_car.shape
        model = QUANTLHModel(m, n)
        model.setAttractorsAj(df_floorspace, 'Zone', 'Residential_FloorSpace')
        model.setPopulationEi(dfEi, 'zone', 'employed')
        model.setCostMatrixCij(cij_car, cij_bus, cij_rail)

        # Run model
        Tij, cbar_k = model.run3modes_NoCalibration(Beta_calibrated)

        # Compute the probability of a flow from a zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities3modes(Tij)

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.
        DjPred_Car = Tij[0].sum(axis=1)
        Ji_Car = Calculate_Job_Accessibility(DjPred_Car, cij_car)

        DjPred_bus = Tij[1].sum(axis=1)
        Ji_bus = Calculate_Job_Accessibility(DjPred_bus, cij_bus)

        DjPred_rail = Tij[2].sum(axis=1)
        Ji_rail = Calculate_Job_Accessibility(DjPred_rail, cij_rail)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'zone': zonecodes_TUR_list, 'JAcar30': Ji_Car, 'JAbus30': Ji_bus, 'JArail30': Ji_rail})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibility2030"])
        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.

        OiPred_Car = Tij[0].sum(axis=0)
        Hi_Car = Calculate_Housing_Accessibility(OiPred_Car, cij_car)

        OiPred_bus = Tij[1].sum(axis=0)
        Hi_bus = Calculate_Housing_Accessibility(OiPred_bus, cij_bus)

        OiPred_rail = Tij[2].sum(axis=0)
        Hi_rail = Calculate_Housing_Accessibility(OiPred_rail, cij_rail)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_TUR_list, 'HAcar30': Hi_Car, 'HAbus30': Hi_bus, 'HArail30': Hi_rail})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibility2030"])

        # Create a Oi Dj table
        dfEi['DjPred_Cars_30'] = Tij[0].sum(axis=1)
        dfEi['DjPred_Bus_30'] = Tij[1].sum(axis=1)
        dfEi['DjPred_Rail_30'] = Tij[2].sum(axis=1)
        dfEi['DjPred_Tot_30'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1) + Tij[2].sum(axis=1)
        dfEi['OiPred_Cars_30'] = Tij[0].sum(axis=0)
        dfEi['OiPred_Bus_30'] = Tij[1].sum(axis=0)
        dfEi['OiPred_Rail_30'] = Tij[2].sum(axis=0)
        dfEi['OiPred_Tot_30'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0) + Tij[2].sum(axis=0)
        dfEi['Job_accessibility_Cars'] = Jobs_accessibility_df['JAcar30']
        dfEi['Jobs_accessibility_bus'] = Jobs_accessibility_df['JAbus30']
        dfEi['Jobs_accessibility_rail'] = Jobs_accessibility_df['JArail30']
        dfEi['Housing_accessibility_Cars'] = Housing_accessibility_df['HAcar30']
        dfEi['Housing_accessibility_bus'] = Housing_accessibility_df['HAbus30']
        dfEi['Housing_accessibility_rail'] = Housing_accessibility_df['HArail30']
        dfEi.to_csv(outputs["JobsDjOi2030"])

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
        # I need my own zone codes file containing the zonei and greek grid indexes as
        # ZoneCodes_ATH does not contain the information
        flow_zonecodes = pd.read_csv(inputs["ZonesCoordinates"])
        flow_car = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(outputs["ArrowsFlowsRoads2030"], 'w') as f:
            dump(flow_car, f)
        flow_bus = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(outputs["ArrowsFlowsBus2030"], 'w') as f:
            dump(flow_bus, f)
        flow_rail = flowArrowsGeoJSON(Tij[2], flow_zonecodes)
        with open(outputs["ArrowsFlowsRail2030"], 'w') as f:
            dump(flow_rail, f)

        print("JtW model", Scenario, "cbar [Cars, bus, rail] = ", cbar_k)

        # Calculate predicted population
        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_TUR_list

        end = time.perf_counter()
        print("Journey to work model run elapsed time (secs)=", end - start)
        print()

        return Beta_calibrated, DjPred

################################################################################
# Schools Model
################################################################################
"""Turin schools:
    # - Primary schools (age 6-10)
    # - Middle schools (age 11-13)
    # - High schools (age 14-18)
    # - Universities (age 18+)
    # Each school type corresponds to a different LUTI model
    # New University Development 2030:
    # 1. Extension of capacity of Politecnico Lingotto 5.000 -> 7.500
    # 2. Extension of capacity of Facolta Agrria et Veterinaria 5.000 -> 10.000
    # 3. Schools maintain their capacity 
"""
def runPrimarySchoolsModel(cij_car_primary, cij_bus_primary, cij_rail_primary,  beta_input, inputs, outputs, Scenario='2019'):

    print("runSchoolsModel running - primary schools")
    start = time.perf_counter()

    # Now schools data:

    primaryZones, primaryAttractors = QUANTSchoolsModel.loadSchoolsData(inputs["PrimaryCapacity2019"])
    row, col = primaryZones.shape
    print("primaryZones count = ", row)

    primaryZones.to_csv(data_primary_zones)
    primaryAttractors.to_csv(data_primary_attractors)

    if Scenario == '2019':
        # Population data:
        PrimaryPopulation = pd.read_csv(inputs["DataSchoolsPupils2019"], usecols=['ZONE', 'PrimarySchool'], index_col='ZONE')
        PrimaryPopulation['ZONE'] = PrimaryPopulation.index  # turn the index (i.e. zone codes) back into a columm
        PrimaryPopulation.reset_index(drop=True,inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        PrimaryPopulation.rename(columns={'ZONE': 'zonei'}, inplace=True)

        # Use cij as cost matrix
        m, n = cij_car_primary.shape
        model = QUANTSchoolsModel(m,n)
        model.setAttractorsAj(primaryAttractors,'zonei','SchoolCapacity')
        model.setPopulationEi(PrimaryPopulation,'zonei','PrimarySchool')
        model.setCostMatrixCij(cij_car_primary, cij_bus_primary, cij_rail_primary)
        beta = beta_input # from the Journey to work model calibration

        # Pij = pupil flows
        primary_Pij, cbar_primary = model.run3modes_NoCalibration(beta)

        print("Primary schools model ", Scenario, " cbar [Cars, bus, rail] = ", cbar_primary)
        # Compute Probabilities:
        primary_probPij = model.computeProbabilities3modes(primary_Pij)

        # Save output matrices
        print("Saving output matrices...")
        # Probabilities:
        np.savetxt(outputs["PrimaryProbPijRoads2019"], primary_probPij[0], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijBus2019"], primary_probPij[1], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijRail2019"], primary_probPij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["PrimaryPijRoads2019"], primary_Pij[0], delimiter=",")
        np.savetxt(outputs["PrimaryPijBus2019"], primary_Pij[1], delimiter=",")
        np.savetxt(outputs["PrimaryPijRail2019"] , primary_Pij[2], delimiter=",")

    elif Scenario == 'NewLandUse&Infr2030':

        # Now schools data:
        PrimaryPopulation_2030 = pd.read_csv(inputs["DataSchoolsPupils2030"], usecols=['ZONE', 'PrimarySchool'], index_col='ZONE')
        PrimaryPopulation_2030['ZONE'] = PrimaryPopulation_2030.index  # turn the index (i.e. zone codes) back into a columm
        PrimaryPopulation_2030.reset_index(drop=True,inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        PrimaryPopulation_2030.rename(columns={'ZONE': 'zonei'}, inplace=True)

        # Use cij as cost matrix
        m, n = cij_car_primary.shape
        model = QUANTSchoolsModel(m, n)
        model.setAttractorsAj(primaryAttractors, 'zonei', 'SchoolCapacity')
        model.setPopulationEi(PrimaryPopulation_2030, 'zonei', 'PrimarySchool')
        model.setCostMatrixCij(cij_car_primary, cij_bus_primary, cij_rail_primary)
        beta = beta_input  # from the Journey to work model calibration

        # Pij = pupil flows
        primary_Pij_30, cbar_primary_30 = model.run3modes_NoCalibration(beta)

        print("Primary schools model ", Scenario, " cbar [Cars, bus, rail] = ", cbar_primary_30)
        # Compute Probabilities:
        primary_probPij_30 = model.computeProbabilities3modes(primary_Pij_30)

         # Save output matrices
        print("Saving output matrices...")
        # Probabilities:
        np.savetxt(outputs["PrimaryProbPijRoads2030"], primary_probPij_30[0], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijBus2030"], primary_probPij_30[1], delimiter=",")
        np.savetxt(outputs["PrimaryProbPijRail2030"], primary_probPij_30[2], delimiter=",")

        # Flows
        np.savetxt(outputs["PrimaryPijRoads2030"], primary_Pij_30[0], delimiter=",")
        np.savetxt(outputs["PrimaryPijBus2030"], primary_Pij_30[1], delimiter=",")
        np.savetxt(outputs["PrimaryPijRail2030"], primary_Pij_30[2], delimiter=",")

        end = time.perf_counter()
        print("Primary school model run elapsed time (secs) = ", end - start)
        print()

def runMiddleSchoolsModel(cij_car_middle, cij_bus_middle, cij_rail_middle, beta_input, inputs, outputs, Scenario='2019'):

    print("runSchoolsModel running - middle schools")
    start = time.perf_counter()

    # Now schools data:

    middleZones, middleAttractors = QUANTSchoolsModel.loadSchoolsData(inputs["MiddleCapacity2019"])
    row, col = middleZones.shape
    print("middleZones count = ", row)

    middleZones.to_csv(data_middle_zones)
    middleAttractors.to_csv(data_middle_attractors)

    if Scenario == '2019':
        # Population data:
        MiddlePopulation = pd.read_csv(inputs["DataSchoolsPupils2019"], usecols=['ZONE', 'MiddleSchool'], index_col='ZONE')
        MiddlePopulation['ZONE'] = MiddlePopulation.index  # turn the index (i.e. zone codes) back into a columm
        MiddlePopulation.reset_index(drop=True,inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        MiddlePopulation.rename(columns={'ZONE': 'zonei'}, inplace=True)

        # Use cij as cost matrix
        m, n = cij_car_middle.shape
        model = QUANTSchoolsModel(m,n)
        model.setAttractorsAj(middleAttractors,'zonei','SchoolCapacity')
        model.setPopulationEi(MiddlePopulation,'zonei','MiddleSchool')
        model.setCostMatrixCij(cij_car_middle, cij_bus_middle, cij_rail_middle)
        beta = beta_input # from the Journey to work model calibration

        # Pij = pupil flows
        middle_Pij, cbar_middle = model.run3modes_NoCalibration(beta)

        print("Middle schools model ", Scenario, " cbar [Cars, bus, rail] = ", cbar_middle)
        # Compute Probabilities:
        middle_probPij = model.computeProbabilities3modes(middle_Pij)

        # Save output matrices
        print("Saving output matrices...")
        # Probabilities:
        np.savetxt(outputs["MiddleProbPijRoads2019"], middle_probPij[0], delimiter=",")
        np.savetxt(outputs["MiddleProbPijBus2019"], middle_probPij[1], delimiter=",")
        np.savetxt(outputs["MiddleProbPijRail2019"], middle_probPij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["MiddlePijRoads2019"], middle_Pij[0], delimiter=",")
        np.savetxt(outputs["MiddlePijBus2019"], middle_Pij[1], delimiter=",")
        np.savetxt(outputs["MiddlePijRail2019"], middle_Pij[2], delimiter=",")

    elif Scenario == 'NewLandUse&Infr2030':
        # Now schools data:
        MiddlePopulation_2030 = pd.read_csv(inputs["DataSchoolsPupils2030"], usecols=['ZONE', 'MiddleSchool'], index_col='ZONE')
        MiddlePopulation_2030['ZONE'] = MiddlePopulation_2030.index  # turn the index (i.e. zone codes) back into a columm
        MiddlePopulation_2030.reset_index(drop=True,inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        MiddlePopulation_2030.rename(columns={'ZONE': 'zonei'}, inplace=True)

        # Use cij as cost matrix
        m, n = cij_car_middle.shape
        model = QUANTSchoolsModel(m, n)
        model.setAttractorsAj(middleAttractors, 'zonei', 'SchoolCapacity')
        model.setPopulationEi(MiddlePopulation_2030, 'zonei', 'MiddleSchool')
        model.setCostMatrixCij(cij_car_middle, cij_bus_middle, cij_rail_middle)
        beta = beta_input  # from the Journey to work model calibration

        # Pij = pupil flows
        middle_Pij_30, cbar_middle_30 = model.run3modes_NoCalibration(beta)

        print("middle schools model ", Scenario, " cbar [Cars, bus, rail] = ", cbar_middle_30)
        # Compute Probabilities:
        middle_probPij_30 = model.computeProbabilities3modes(middle_Pij_30)

         # Save output matrices
        print("Saving output matrices...")
        # Probabilities:
        np.savetxt(outputs["MiddleProbPijRoads2030"], middle_probPij_30[0], delimiter=",")
        np.savetxt(outputs["MiddleProbPijBus2030"], middle_probPij_30[1], delimiter=",")
        np.savetxt(outputs["MiddleProbPijRail2030"], middle_probPij_30[2], delimiter=",")

        # Flows
        np.savetxt(outputs["MiddlePijRoads2030"], middle_Pij_30[0], delimiter=",")
        np.savetxt(outputs["MiddlePijBus2030"], middle_Pij_30[1], delimiter=",")
        np.savetxt(outputs["MiddlePijRail2030"], middle_Pij_30[2], delimiter=",")

        end = time.perf_counter()
        print("Middle school model run elapsed time (secs) = ", end - start)
        print()

def runHighSchoolsModel(cij_car_high, cij_bus_high, cij_rail_high, beta_input, inputs, outputs, Scenario='2019'):

        print("runSchoolsModel running - high schools")
        start = time.perf_counter()

        # Now schools data:

        highZones, highAttractors = QUANTSchoolsModel.loadSchoolsData(inputs["HighCapacity2019"])
        row, col = highZones.shape
        print("highZones count = ", row)

        highZones.to_csv(data_high_zones)
        highAttractors.to_csv(data_high_attractors)

        if Scenario == '2019':

            # Population data:
            highPopulation = pd.read_csv(inputs["DataSchoolsPupils2019"], usecols=['ZONE', 'HighSchool'], index_col='ZONE')
            highPopulation['ZONE'] = highPopulation.index  # turn the index (i.e. zone codes) back into a columm
            highPopulation.reset_index(drop=True,inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
            highPopulation.rename(columns={'ZONE': 'zonei'}, inplace=True)

            # Use cij as cost matrix
            m, n = cij_car_high.shape
            model = QUANTSchoolsModel(m, n)
            model.setAttractorsAj(highAttractors, 'zonei', 'SchoolCapacity')
            model.setPopulationEi(highPopulation, 'zonei', 'HighSchool')
            model.setCostMatrixCij(cij_car_high, cij_bus_high, cij_rail_high)
            beta = beta_input  # from the Journey to work model calibration

            # Pij = pupil flows
            high_Pij, cbar_high = model.run3modes_NoCalibration(beta)

            print("High schools model ", Scenario, " cbar [Cars, bus, rail] = ", cbar_high)
            # Compute Probabilities:
            high_probPij = model.computeProbabilities3modes(high_Pij)

            # Save output matrices
            print("Saving output matrices...")
            # Probabilities:
            np.savetxt(outputs["HighProbPijRoads2019"], high_probPij[0], delimiter=",")
            np.savetxt(outputs["HighProbPijBus2019"], high_probPij[1], delimiter=",")
            np.savetxt(outputs["HighProbPijRail2019"], high_probPij[2], delimiter=",")

            # Flows
            np.savetxt(outputs["HighPijRoads2019"], high_Pij[0], delimiter=",")
            np.savetxt(outputs["HighPijBus2019"], high_Pij[1], delimiter=",")
            np.savetxt(outputs["HighPijRail2019"], high_Pij[2], delimiter=",")

        elif Scenario == 'NewLandUse&Infr2030':
            # Now schools data:
            highPopulation_2030 = pd.read_csv(inputs["DataSchoolsPupils2030"], usecols=['ZONE', 'HighSchool'], index_col='ZONE')
            highPopulation_2030['ZONE'] = highPopulation_2030.index  # turn the index (i.e. zone codes) back into a columm
            highPopulation_2030.reset_index(drop=True,inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
            highPopulation_2030.rename(columns={'ZONE': 'zonei'}, inplace=True)

            # Use cij as cost matrix
            m, n = cij_car_high.shape
            model = QUANTSchoolsModel(m, n)
            model.setAttractorsAj(highAttractors, 'zonei', 'SchoolCapacity')
            model.setPopulationEi(highPopulation_2030, 'zonei', 'HighSchool')
            model.setCostMatrixCij(cij_car_high, cij_bus_high, cij_rail_high)
            beta = beta_input  # from the Journey to work model calibration

            # Pij = pupil flows
            high_Pij_30, cbar_high_30 = model.run3modes_NoCalibration(beta)

            print("high schools model ", Scenario, " cbar [Cars, bus, rail] = ", cbar_high_30)
            # Compute Probabilities:
            high_probPij_30 = model.computeProbabilities3modes(high_Pij_30)

            # Save output matrices
            print("Saving output matrices...")
            # Probabilities:
            np.savetxt(outputs["HighProbPijRoads2030"], high_probPij_30[0], delimiter=",")
            np.savetxt(outputs["HighProbPijBus2030"], high_probPij_30[1], delimiter=",")
            np.savetxt(outputs["HighProbPijRail2030"], high_probPij_30[2], delimiter=",")

            # Flows
            np.savetxt(outputs["HighPijRoads2030"], high_Pij_30[0], delimiter=",")
            np.savetxt(outputs["HighPijBus2030"], high_Pij_30[1], delimiter=",")
            np.savetxt(outputs["HighPijRail2030"], high_Pij_30[2], delimiter=",")

            end = time.perf_counter()
            print("High school model run elapsed time (secs) = ", end - start)
            print()

def runUniversitiesModel(cij_car_uni, cij_bus_uni, cij_rail_uni, beta_input, inputs, outputs, Scenario='2019'):
    print("runSchoolsModel running - Universities")
    start = time.perf_counter()

    if Scenario == '2019':
        # Population data:
        uniPopulation = pd.read_csv(inputs["DataUniStudents2019"], usecols=['ZONE', 'University_Students_2019'], index_col='ZONE')
        uniPopulation['ZONE'] = uniPopulation.index  # turn the index (i.e. zone codes) back into a columm
        uniPopulation.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        uniPopulation.rename(columns={'ZONE': 'zonei'}, inplace=True)

        # Now schools data:
        uniZones, uniAttractors = QUANTSchoolsModel.loadSchoolsData(inputs["UniCapacity2019"])
        row, col = uniZones.shape
        print("uniZones count = ", row)

        uniZones.to_csv(data_unis_zones)
        uniAttractors.to_csv(data_unis_attractors)

        # Use cij as cost matrix
        m, n = cij_car_uni.shape
        model = QUANTSchoolsModel(m, n)
        model.setAttractorsAj(uniAttractors, 'zonei', 'SchoolCapacity')
        model.setPopulationEi(uniPopulation, 'zonei', 'University_Students_2019')
        model.setCostMatrixCij(cij_car_uni, cij_bus_uni, cij_rail_uni)
        beta = beta_input  # from the Journey to work model calibration

        # Pij = pupil flows
        uni_Pij, cbar_uni = model.run3modes_NoCalibration(beta)

        print("Universities model ", Scenario, " cbar [Car, bus, rail] = ", cbar_uni)
        # Compute Probabilities:
        uni_probPij = model.computeProbabilities3modes(uni_Pij)

        # Save output matrices
        print("Saving output matrices...")
        # Probabilities:
        np.savetxt(outputs["UniProbPijRoads2019"], uni_probPij[0], delimiter=",")
        np.savetxt(outputs["UniProbPijBus2019"], uni_probPij[1], delimiter=",")
        np.savetxt(outputs["UniProbPijRail2019"], uni_probPij[2], delimiter=",")

        # Flows
        np.savetxt(outputs["UniPijRoads2019"], uni_Pij[0], delimiter=",")
        np.savetxt(outputs["UniPijBus2019"], uni_Pij[1], delimiter=",")
        np.savetxt(outputs["UniPijRail2019"], uni_Pij[2], delimiter=",")

    elif Scenario == 'NewLandUse&Infr2030':
        # Now schools data:
        uniPopulation_2030 = pd.read_csv(inputs["DataUniStudents2030"], usecols=['ZONE', 'University_Students_2030'], index_col='ZONE')
        uniPopulation_2030['ZONE'] = uniPopulation_2030.index  # turn the index (i.e. zone codes) back into a columm
        uniPopulation_2030.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        uniPopulation_2030.rename(columns={'ZONE': 'zonei'}, inplace=True)

        uniZones_30, uniAttractors_30 = QUANTSchoolsModel.loadSchoolsData(inputs["UniCapacity2030"])
        row, col = uniZones_30.shape
        print("uniZones for 2030 count = ", row)

        uniZones_30.to_csv(data_unis_zones_2030)
        uniAttractors_30.to_csv(data_unis_attractors_2030)

        # Use cij as cost matrix
        m, n = cij_car_uni.shape
        model = QUANTSchoolsModel(m, n)
        model.setAttractorsAj(uniAttractors_30, 'zonei', 'SchoolCapacity')
        model.setPopulationEi(uniPopulation_2030, 'zonei', 'University_Students_2030')
        model.setCostMatrixCij(cij_car_uni, cij_bus_uni, cij_rail_uni)
        beta = beta_input  # from the Journey to work model calibration

        # Pij = pupil flows
        uni_Pij_30, cbar_uni_30 = model.run3modes_NoCalibration(beta)

        print("Universities model ", Scenario, " cbar [Cars, bus, rail] = ", cbar_uni_30)
        # Compute Probabilities:
        uni_probPij_30 = model.computeProbabilities3modes(uni_Pij_30)

        # Save output matrices
        print("Saving output matrices...")
        # Probabilities:
        np.savetxt(outputs["UniProbPijRoads2030"], uni_probPij_30[0], delimiter=",")
        np.savetxt(outputs["UniProbPijBus2030"], uni_probPij_30[1], delimiter=",")
        np.savetxt(outputs["UniProbPijRail2030"], uni_probPij_30[2], delimiter=",")

        # Flows
        np.savetxt(outputs["UniPijRoads2030"], uni_Pij_30[0], delimiter=",")
        np.savetxt(outputs["UniPijBus2030"], uni_Pij_30[1], delimiter=",")
        np.savetxt(outputs["UniPijRail2030"], uni_Pij_30[2], delimiter=",")

        end = time.perf_counter()
        print("Universities model run elapsed time (secs) = ", end - start)
        print()


################################################################################
# Hospitals Model                                                              #
################################################################################

def runHospitalsModel(cij_car_hosp, cij_bus_hosp, cij_rail_hosp, beta_input, inputs, outputs, Scenario='2019'):
    print("Running Hospitals model")
    start = time.perf_counter()
    # Hospitals model
    # New Land Use and Infrastructure 2030:
    # 1. New Hospital (Citta de la Salute) in zone no. 618
    # 2. Demolish 4 hospitals [a. Azienda Ospendaliera O.I.R.M.S. Sant’ Anna (zone no. 552), b. Ospedale Molinette (zone no. 551), c. Ospedale Maggiore (zone no. 1014), d. Ospedale Santa Croce (zone no. 1008)]
    # 3. Remove beds from 28 (158), 29 (181) and move them to Casa di cura villa de salute (zone no. 1026)
    # load hospitals population
    # load jobs data for residential zones


    if Scenario == '2019':
        hospitalPopulation = pd.read_csv(inputs["DataPopulation2019"], usecols=['ZONE', 'Pop'], index_col='ZONE')
        hospitalPopulation['ZONE'] = hospitalPopulation.index  # turn the index (i.e. zone codes) back into a columm
        hospitalPopulation.reset_index(drop=True,inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        hospitalPopulation.rename(columns={'ZONE': 'zonei'}, inplace=True)

        # load hospitals population
        hospitalZones, hospitalAttractors = QUANTHospitalsModel.loadHospitalsData(inputs["DataHospitals2019"])

        row, col = hospitalZones.shape
        print("hospitalZones count = ", row)

        hospitalZones.to_csv(data_hospital_zones)
        hospitalAttractors.to_csv(data_hospital_attractors)

        # Use cij as cost matrix
        m, n = cij_bus_hosp.shape
        model = QUANTHospitalsModel(m, n)
        model.setAttractorsAj(hospitalAttractors, 'zonei', 'Number_of_beds')
        model.setPopulationEi(hospitalPopulation, 'zonei', 'Pop')
        model.setCostMatrixCij(cij_car_hosp, cij_bus_hosp, cij_rail_hosp)
        beta = beta_input  # from the Journey to work model calibration

        # Hij = hospital flows
        hospital_Hij, cbar = model.run3modes_NoCalibration(beta)

        print("Hospitals model ", Scenario, " cbar [Car, bus, rail] = ", cbar)

        # Compute Probabilities:
        hospital_probHij = model.computeProbabilities3modes(hospital_Hij)

        # Save output matrices
        print("Saving output matrices...")

        # Probabilities:
        np.savetxt(outputs["HospitalsProbPijRoads2019"], hospital_probHij[0], delimiter=",")
        np.savetxt(outputs["HospitalsProbPijBus2019"], hospital_probHij[1], delimiter=",")
        np.savetxt(outputs["HospitalsProbPijRail2019"], hospital_probHij[2], delimiter=",")
        # Flows
        np.savetxt(outputs["HospitalsPijRoads2019"], hospital_Hij[0], delimiter=",")
        np.savetxt(outputs["HospitalsPijBus2019"], hospital_Hij[1], delimiter=",")
        np.savetxt(outputs["HospitalsPijRail2019"], hospital_Hij[2], delimiter=",")

    elif Scenario == 'NewLandUse&Infr2030':
        hospitalPopulation_30 = pd.read_csv(inputs["DataPopulation2030"], usecols=['ZONE', 'Pop_2030'], index_col='ZONE')
        hospitalPopulation_30['ZONE'] = hospitalPopulation_30.index  # turn the index (i.e. zone codes) back into a columm
        hospitalPopulation_30.reset_index(drop=True, inplace=True)  # IMPORTANT, otherwise indexes remain for ALL the rows i.e. idx=0..OriginalN NOT true row count!
        hospitalPopulation_30.rename(columns={'ZONE': 'zonei'}, inplace=True)
        # load hospitals population
        hospitalZones30, hospitalAttractors30 = QUANTHospitalsModel.loadHospitalsData(inputs["DataHospitals2030"])

        row, col = hospitalZones30.shape
        print("hospitalZones_2030 count = ", row)

        hospitalZones30.to_csv(data_hospital_zones_2030)
        hospitalAttractors30.to_csv(data_hospital_attractors_2030)

        # Use cij as cost matrix
        m, n = cij_bus_hosp.shape
        model = QUANTHospitalsModel(m, n)
        model.setAttractorsAj(hospitalAttractors30, 'zonei', 'Number_of_beds')
        model.setPopulationEi(hospitalPopulation_30, 'zonei', 'Pop_2030')
        model.setCostMatrixCij(cij_car_hosp, cij_bus_hosp, cij_rail_hosp)
        beta = beta_input  # from the Journey to work model calibration

        # Hij = hospital flows
        hospital_Hij_30, cbar = model.run3modes_NoCalibration(beta)

        print("Hospitals model ", Scenario, " cbar [Car, bus, rail] = ", cbar)

        # Compute Probabilities:
        hospital_probHij_30 = model.computeProbabilities3modes(hospital_Hij_30)

        # Save output matrices
        print("Saving output matrices...")

        # Probabilities:
        np.savetxt(outputs["HospitalsProbPijRoads2030"], hospital_probHij_30[0], delimiter=",")
        np.savetxt(outputs["HospitalsProbPijBus2030"], hospital_probHij_30[1], delimiter=",")
        np.savetxt(outputs["HospitalsProbPijRail2030"], hospital_probHij_30[2], delimiter=",")

        # Flows
        np.savetxt(outputs["HospitalsPijRoads2030"], hospital_Hij_30[0], delimiter=",")
        np.savetxt(outputs["HospitalsPijBus2030"], hospital_Hij_30[1], delimiter=",")
        np.savetxt(outputs["HospitalsPijRail2030"], hospital_Hij_30[2], delimiter=",")

    end = time.perf_counter()
    print("hospitals model run", Scenario, " elapsed time (secs) = ", end - start)
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
    # Housing accessibility is the distribution of population around a job location.
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


