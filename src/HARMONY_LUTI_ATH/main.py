"""
HARMONY Land-Use Transport-Interaction Model - Athens case study
main.py

Author: Fulvio D. Lopane, Centre for Advanced Spatial Analysis, University College London
https://www.casa.ucl.ac.uk

- Developed from Richard Milton's QUANT_RAMP
- Further developed from Eleni Kalantzi's code for MSc dissertation
Msc Smart Cities and Urban Analytics, Centre for Advanced Spatial Analysis, University College London
"""

import time
from geojson import dump
import numpy as np
import pandas as pd
import os

from HARMONY_LUTI_ATH.globals import *
from HARMONY_LUTI_ATH.analytics import graphProbabilities, flowArrowsGeoJSON
from HARMONY_LUTI_ATH.quantlhmodel import QUANTLHModel
from HARMONY_LUTI_ATH.maps import *

import csv

def start_main(inputs, outputs, logger):
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

    ################################################################################
    # Now on to the model run section
    ################################################################################

    # Zone Codes for Athens
    logger.warning("Importing ATH cij matrices")

    zonecodes_ATH = pd.read_csv(inputs["ZoneCodesFile"])
    zonecodes_ATH.set_index('zone')
    zonecodes_ATH_list = zonecodes_ATH['zone'].tolist()

    dfcoord = pd.read_csv(inputs["ZoneCoordinates"], usecols=['zone', 'Greek_Grid_east', 'Greek_Grid_north'], index_col='zone')

    # Import cij data

    # Fisrt, import intrazone distances
    intrazone_dist_df = pd.read_csv(inputs["IntrazoneDist"], usecols=["Intrazone_dist"])  # create df from csv
    intrazone_dist_list = intrazone_dist_df.Intrazone_dist.values.tolist()  # save the intrazone distances into a list

    # Import the csv cij private as Pandas DataFrame:
    cij_pr_df = pd.read_csv(inputs["CijPrivateMinFilename"], header=None)

    # Convert the dataframe to a numpy matrix:
    cij_pr = cij_pr_df.to_numpy()
    cij_pr[cij_pr < 0] = 120  # upper limit of two hours - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_pr[cij_pr < 1] = 1  # lower limit of 1 minute links

    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_pr = 13  # define average speed in km/h - based on Athens vehicles
    average_TT_pr = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_pr.append((i / 1000) / (av_speed_pr / 60))  # result in minutes
    np.fill_diagonal(cij_pr, average_TT_pr)  # save the average TT list as the main diagonal of the cij matrix

    # Print the dimensions of the matrix to check that everything is ok:
    logger.warning('cij private shape: ' + str(cij_pr.shape))
    cij_pr[cij_pr < 1] = 1  # repeat of above after leading diagonal zeros added
    cij_pr = cij_pr * 2.5  # HACK! Make driving less attractive - are costs wrong? --> probably speeds from OASA not to be trusted?

    # Import the csv cij public as Pandas DataFrame:
    cij_pu_df = pd.read_csv(inputs["CijPublicMinFilename"], header=None)

    # Convert the dataframe to a nupmy matrix:
    cij_pu = cij_pu_df.to_numpy()
    cij_pu[cij_pu < 0] = 120  # upper limit of two hours - change missing vlaues (set as -1) to very high numbers to simulate impossible routes
    cij_pu[cij_pu < 1] = 1  # lower limit of 1 minute links
    # Add waiting time: 16min for Athen from Moovit data:
    waiting_time_proportion = 16.0 / 47.0  # 16 = average waiting time; 47 = average travel time
    cij_pu[cij_pu < 9999] *= (1 + waiting_time_proportion)

    # Change values in the main diagonal from 0 to intrazone impedance:
    av_speed_pu = 10  # define average speed in km/h - based on Athens buses
    average_TT_pu = []  # average intrazonal travel time
    for i in intrazone_dist_list:
        average_TT_pu.append((i / 1000) / (av_speed_pu / 60))  # result in minutes
    np.fill_diagonal(cij_pu, average_TT_pu)  # save the average TT list as the main diagonal of the cij matrix

    # Print the dimensions of the matrix to check that everything is ok:
    logger.warning('cij public shape: ' + str(cij_pu.shape))
    # _____________________________________________________________________________________
    '''
    # Import OD data
    # Import the csv OD private as Pandas DataFrame:
    # OD_pr_df = pd.read_csv(ODPrFilename_csv, header=None)
    # Convert the dataframe to a numpy matrix:
    # OD_pr = OD_pr_df.to_numpy()
    # print('OD private shape: ', OD_pr.shape)

    # Import the csv OD public as Pandas DataFrame:
    # OD_pu_df = pd.read_csv(ODPuFilename_csv, header=None)
    # Convert the dataframe to a numpy matrix:
    # OD_pu = OD_pu_df.to_numpy()
    # print('OD public shape: ', OD_pu.shape)
    '''
    # _____________________________________________________________________________________

    # now run the relevant models to produce the outputs
    runEllinikoScenarios(cij_pr, cij_pu, zonecodes_ATH_list, inputs, outputs, logger)

    # Population maps:
    population_map_creation(inputs, outputs, logger)

    # Flows maps:
    # THIS FEATURE IS TURNED OFF - long run time - only for HQ flows visualisation
    create_flow_maps = False
    if create_flow_maps:
        flows_output_keys = ["JobsTijPublic2019", "JobsTijPrivate2019", "JobsTijPublic2030", "JobsTijPrivate2030", "JobsTijPublic2045", "JobsTijPrivate2045"]
        flows_map_creation(inputs, outputs, flows_output_keys)

################################################################################
def runEllinikoScenarios(cij_pr, cij_pu, zonecodes_ATH_list, inputs, outputs, logger):
    # First run the base model to calibrate it with 2011 observed trip data:
    # Run Journey to work model:
    beta_2019, DjPred_JtW_2019 = runJourneyToWorkModel(cij_pr, cij_pu, zonecodes_ATH_list, inputs, outputs, logger)

    # Now run the JtW model with 2011 beta and 2019 pop (without calibration this time)
    DjPred_JtW_2030 = runJourneyToWorkModel(cij_pr, cij_pu, zonecodes_ATH_list, inputs, outputs, logger, 'Elliniko_2030', beta_2019)
    DjPred_JtW_2045 = runJourneyToWorkModel(cij_pr, cij_pu, zonecodes_ATH_list, inputs, outputs, logger, 'Elliniko_2045', beta_2019)

################################################################################
# Journey to work Model                                                        #
################################################################################

"""
runJourneyToWorkModel
"""
# Journey to work model with households (HH) floorspace as attractor
def runJourneyToWorkModel(cij_pr, cij_pu, zonecodes_ATH_list, inputs, outputs, logger, Scenario='2019', Beta_calibrated=None):
    logger.warning("Running Journey to Work " + str(Scenario) + " model.")
    start = time.perf_counter()
    # Singly constrained model:
    # We conserve the number of jobs and predict the working population residing in Athens zones
    # journeys to work generated by jobs
    # Origins: workplaces
    # Destinations: Zones' households
    # Attractor: floorspace of housing

    """
                    Journey to work       |   Retail model
     Origins:       workplaces            |   households
     Destinations:  households            |   supermarkets
     conserved:     jobs                  |   income
     predicted:     population of zones   |   expenditure @ supermarkets
     attractor:     HH floorspace density |   supermarket floorspace
    """
    dfcoord = pd.read_csv(inputs["ZoneCoordinates"], usecols=['zone', 'Greek_Grid_east', 'Greek_Grid_north'], index_col='zone')
    # load jobs data for residential zones
    dfEi = pd.read_csv(inputs["DataEmployment2019"], usecols=['zone','employment'], index_col='zone')
    dfEi.astype({'employment': 'int64'})

    # df_pop = pd.read_csv(data_census_pop, usecols=['zone','pop_tot'], index_col='zone')

    df_floorspace = pd.read_csv(inputs["HhFloorspace2019"], usecols=['zone','hh_floorspace_density'], index_col='zone')
    # Need to substitute 0 values in floorspace dataframe with very low values (e.g. 1) to avoid division by zero:
    df_floorspace.replace(0, 1, inplace=True)

    # if we are running a scenario, update the Zones with the new attractors and the numbers of jobs
    if Scenario == '2019':
        # This is the base scenario, so we don't have to modify the attractors/jobs
        pass

    elif Scenario == 'Elliniko_2030':
        dfEi_30 = pd.read_csv(inputs["DataEmployment2030"], usecols=['zone', 'employment'], index_col='zone')

        # df_pop = pd.read_csv(data_census_pop, usecols=['zone', 'pop_tot'], index_col='zone')

        df_floorspace = pd.read_csv(inputs["HhFloorspace2019"], usecols=['zone', 'hh_floorspace_density'], index_col='zone')
        # Need to substitute 0 values in floorspace dataframe with very low values (e.g. 1) to avoid division by zero:
        df_floorspace.replace(0, 1, inplace=True)

    elif Scenario == 'Elliniko_2045':
        dfEi_45 = pd.read_csv(inputs["DataEmployment2045"], usecols=['zone', 'employment'], index_col='zone')
        dfEi_45.astype({'employment': 'int64'})

        # Scenario_pop_table = pd.read_csv(pop_2045, usecols=['zone', 'pop_tot'], index_col='zone')

        Scenario_floorspace = pd.read_csv(inputs["HhFloorspace2045"], usecols=['zone', 'hh_floorspace_density'], index_col='zone')
        # Need to substitute 0 values in floorspace dataframe with very low values (e.g. 1) to avoid division by zero:
        Scenario_floorspace.replace(0, 1, inplace=True)

    # threshold = 2.7 *(1./1265)  # The threshold below which to ignore low probability trips - for geojson flows files (lines)

    if Scenario == '2019':
        # Load observed data for model calibration:
        # OD_Pu, OD_Pr = QUANTLHModel.loadODData()

        # Use cij as cost matrix (zone to zone)
        m, n = cij_pu.shape
        model = QUANTLHModel(m, n)

        # OBS Cbar public is 47 minutes, according to Public transit facts & statistics for Athens | Moovit Public Transit Index (moovitapp.com)
        # OBS Cbar private is 37 minutes, according to Traffic in Athens (numbeo.com)
        model.setObsCbar(47, 37)

        # model.setODMatrix(OD_Pu, OD_Pr)
        model.setAttractorsAj(df_floorspace, 'zone', 'hh_floorspace_density')
        model.setPopulationEi(dfEi, 'zone', 'employment')
        model.setCostMatrixCij(cij_pu, cij_pr)

        Tij, beta_k, cbar_k_pred, cbar_k_obs = model.runTwoModes(logger)

        # Save output matrices
        logger.warning("Saving output matrices...")

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_pu = Tij[0].sum(axis=1)
        Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

        DjPred_pr = Tij[1].sum(axis=1)
        Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'zone': zonecodes_ATH_list, 'JobsApu19': Ji_pu, 'JobsApr19': Ji_pr})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibility2019"])

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        OiPred_pu = Tij[0].sum(axis=0)
        Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

        OiPred_pr = Tij[1].sum(axis=0)
        Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_ATH_list,'HApu19': Hi_pu, 'HApr19': Hi_pr})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibility2019"])

        # NOTE: these are saved later as csvs, but not with an easy to read formatter
        # np.savetxt("debug_Tij_public_2019.txt", Tij[0], delimiter=",", fmt="%i")
        # np.savetxt("debug_Tij_private_2019.txt", Tij[1], delimiter=",", fmt="%i")
        # now an Oj Dj table
        # DfEi is employment - really hope these come out in the right order
        dfEi['DjPred_pu'] = Tij[0].sum(axis=1)
        dfEi['DjPred_pr'] = Tij[1].sum(axis=1)
        dfEi['DjPred'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1)
        dfEi['OiPred_pu'] = Tij[0].sum(axis=0)
        dfEi['OiPred_pr'] = Tij[1].sum(axis=0)
        dfEi['OiPred_19'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0)
        dfEi['Job_accessibility_pu'] = Jobs_accessibility_df['JobsApu19']
        dfEi['Job_accessibility_pr'] = Jobs_accessibility_df['JobsApr19']
        dfEi['Housing_accessibility_pu'] = Housing_accessibility_df['HApu19']
        dfEi['Housing_accessibility_pr'] = Housing_accessibility_df['HApr19']
        dfEi['Latitude'] = dfcoord['Greek_Grid_east']
        dfEi['Longitude'] = dfcoord['Greek_Grid_north']
        dfEi.to_csv(outputs["EjOi2019"])

        # print("Computing probabilities...")
        # Compute the probability of a flow from a zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities2modes(Tij)

        # Probabilities:
        np.savetxt(outputs["JobsProbTijPublic2019"], jobs_probTij[0], delimiter=",")
        np.savetxt(outputs["JobsProbTijPrivate2019"], jobs_probTij[1], delimiter=",")

        # People flows
        np.savetxt(outputs["JobsTijPublic2019"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsTijPrivate2019"], Tij[1], delimiter=",")

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and greek grid indexes as
        # ZoneCodes_ATH does not contain the information
        flow_zonecodes = pd.read_csv(inputs["ZoneCoordinates"])
        flow_pu = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(outputs["ArrowsFlowsPublic2019"], 'w') as f:
            dump(flow_pu, f)
        flow_pr = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(outputs["ArrowsFlowsPrivate2019"], 'w') as f:
            dump(flow_pr, f)

        # cbar = model.computeCBar(Tij, cij_road_EW)
        logger.warning("JtW model" + str(Scenario) + "observed cbar [public, private] = " + str(cbar_k_obs))
        logger.warning("JtW model" + str(Scenario) + "predicted cbar [public, private] = " + str(cbar_k_pred))
        logger.warning("JtW model" + str(Scenario) + "beta [public, private] = " + str(beta_k))

        # Calculate predicted population
        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_ATH_list

        end = time.perf_counter()
        # print("Journey to work model", Scenario," run elapsed time (secs)=", end - start)
        logger.warning("Journey to work model run elapsed time (minutes)=" + str((end - start) / 60))

        return beta_k, DjPred

    elif Scenario == 'Elliniko_2030':
        # Use cij as cost matrix
        m, n = cij_pu.shape
        model = QUANTLHModel(m, n)
        model.setAttractorsAj(df_floorspace, 'zone', 'hh_floorspace_density')
        model.setPopulationEi(dfEi_30, 'zone', 'employment')
        model.setCostMatrixCij(cij_pu, cij_pr)

        Tij, cbar_k = model.run2modes_NoCalibration(Beta_calibrated)
        # Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities2modes(Tij)

        # Save output matrices
        logger.warning("Saving output matrices...")

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the population around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_pu = Tij[0].sum(axis=1)
        Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

        DjPred_pr = Tij[1].sum(axis=1)
        Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'zone': zonecodes_ATH_list, 'JobsApu30': Ji_pu, 'JobsApr30': Ji_pr})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibility2030"])

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the jobs around a zone divided by the travel time squared.

        OiPred_pu = Tij[0].sum(axis=0)
        Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

        OiPred_pr = Tij[1].sum(axis=0)
        Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_ATH_list, 'HApu30': Hi_pu, 'HApr30': Hi_pr})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibility2030"])

        # NOTE: these are saved later as csvs, but not with an easy to read formatter
        # np.savetxt("debug_Tij_public_2030.txt", Tij[0], delimiter=",", fmt="%i")
        # np.savetxt("debug_Tij_private_2030.txt", Tij[1], delimiter=",", fmt="%i")
        # now an Oj Dj table
        # DfEi is employment - really hope these come out in the right order
        dfEi['DjPred_pu'] = Tij[0].sum(axis=1)
        dfEi['DjPred_pr'] = Tij[1].sum(axis=1)
        dfEi['DjPred'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1)
        dfEi['OiPred_pu'] = Tij[0].sum(axis=0)
        dfEi['OiPred_pr'] = Tij[1].sum(axis=0)
        dfEi['OiPred_30'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0)
        dfEi['Job_accessibility_pu'] = Jobs_accessibility_df['JobsApu30']
        dfEi['Job_accessibility_pr'] = Jobs_accessibility_df['JobsApr30']
        dfEi['Housing_accessibility_pu'] = Housing_accessibility_df['HApu30']
        dfEi['Housing_accessibility_pr'] = Housing_accessibility_df['HApr30']
        dfEi['Latitude'] = dfcoord['Greek_Grid_east']
        dfEi['Longitude'] = dfcoord['Greek_Grid_north']
        dfEi.to_csv(outputs["EjOi2030"])

        # Probabilities:
        np.savetxt(outputs["JobsProbTijPublic2030"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsProbTijPrivate2030"], Tij[1], delimiter=",")

        # People flows
        np.savetxt(outputs["JobsTijPublic2030"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsTijPrivate2030"], Tij[1], delimiter=",")

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and greek grid indexes as
        # ZoneCodes_ATH does not contain the information
        flow_zonecodes = pd.read_csv(inputs["ZoneCoordinates"])
        flow_pu = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(outputs["ArrowsFlowsPublic2030"], 'w') as f:
            dump(flow_pu, f)
        flow_pr = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(outputs["ArrowsFlowsPrivate2030"], 'w') as f:
            dump(flow_pr, f)

        logger.warning("JtW model" + str(Scenario) + " cbar [public, private] = " + str(cbar_k))

        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_ATH_list

        end = time.perf_counter()
        # print("Journey to work model run elapsed time (secs) =", end - start)
        logger.warning("Journey to work model run elapsed time (minutes) =" + str((end - start)/60))

        return DjPred

    elif Scenario == 'Elliniko_2045':
        # Use cij as cost matrix (zone to zone)
        m, n = cij_pu.shape
        model = QUANTLHModel(m, n)
        model.setAttractorsAj(Scenario_floorspace, 'zone', 'hh_floorspace_density')
        model.setPopulationEi(dfEi_45, 'zone', 'employment')
        model.setCostMatrixCij(cij_pu, cij_pr)

        Tij, cbar_k = model.run2modes_NoCalibration(Beta_calibrated)
        # Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones.
        jobs_probTij = model.computeProbabilities2modes(Tij)

        # Save output matrices
        logger.warning("Saving output matrices...")

        # Jobs accessibility:
        # Job accessibility is the distribution of population around a job location.
        # It’s just the sum of all the jobs around a job zone divided by the travel time squared.
        # This is scaled so that the total of all i zones comes to 100.

        DjPred_pu = Tij[0].sum(axis=1)
        Ji_pu = Calculate_Job_Accessibility(DjPred_pu, cij_pu)

        DjPred_pr = Tij[1].sum(axis=1)
        Ji_pr = Calculate_Job_Accessibility(DjPred_pr, cij_pr)

        # Save output:
        Jobs_accessibility_df = pd.DataFrame({'zone': zonecodes_ATH_list, 'JobsApu45': Ji_pu, 'JobsApr45': Ji_pr})
        Jobs_accessibility_df.to_csv(outputs["JobsAccessibility2045"])

        # Housing Accessibility:
        # Housing accessibility is the distribution of jobs around a housing location.
        # It’s just the sum of all the population around a zone divided by the travel time squared.

        OiPred_pu = Tij[0].sum(axis=0)
        Hi_pu = Calculate_Housing_Accessibility(OiPred_pu, cij_pu)

        OiPred_pr = Tij[1].sum(axis=0)
        Hi_pr = Calculate_Housing_Accessibility(OiPred_pr, cij_pr)

        # Save output:
        Housing_accessibility_df = pd.DataFrame({'zone': zonecodes_ATH_list, 'HApu45': Hi_pu, 'HApr45': Hi_pr})
        Housing_accessibility_df.to_csv(outputs["HousingAccessibility2045"])

        # NOTE: these are saved later as csvs, but not with an easy to read formatter
        # np.savetxt("debug_Tij_public_2045.txt", Tij[0], delimiter=",", fmt="%i")
        # np.savetxt("debug_Tij_private_2045.txt", Tij[1], delimiter=",", fmt="%i")
        # now an Oj Dj table
        # DfEi is employment - really hope these come out in the right order
        dfEi['DjPred_pu'] = Tij[0].sum(axis=1)
        dfEi['DjPred_pr'] = Tij[1].sum(axis=1)
        dfEi['DjPred'] = Tij[0].sum(axis=1) + Tij[1].sum(axis=1)
        dfEi['OiPred_pu'] = Tij[0].sum(axis=0)
        dfEi['OiPred_pr'] = Tij[1].sum(axis=0)
        dfEi['OiPred_45'] = Tij[0].sum(axis=0) + Tij[1].sum(axis=0)
        dfEi['Job_accessibility_pu'] = Jobs_accessibility_df['JobsApu45']
        dfEi['Job_accessibility_pr'] = Jobs_accessibility_df['JobsApr45']
        dfEi['Housing_accessibility_pu'] = Housing_accessibility_df['HApu45']
        dfEi['Housing_accessibility_pr'] = Housing_accessibility_df['HApr45']
        dfEi['Latitude'] = dfcoord['Greek_Grid_east']
        dfEi['Longitude'] = dfcoord['Greek_Grid_north']
        dfEi.to_csv(outputs["EjOi2045"])

        # Probabilities:
        np.savetxt(outputs["JobsProbTijPublic2045"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsProbTijPrivate2045"], Tij[1], delimiter=",")

        # People flows
        np.savetxt(outputs["JobsTijPublic2045"], Tij[0], delimiter=",")
        np.savetxt(outputs["JobsTijPrivate2045"], Tij[1], delimiter=",")

        # Geojson flows files - arrows
        # I need my own zone codes file containing the zonei and greek grid indexes as
        # ZoneCodes_ATH does not contain the information
        flow_zonecodes = pd.read_csv(inputs["ZoneCoordinates"])
        flow_pu = flowArrowsGeoJSON(Tij[0], flow_zonecodes)
        with open(outputs["ArrowsFlowsPublic2045"], 'w') as f:
            dump(flow_pu, f)
        flow_pr = flowArrowsGeoJSON(Tij[1], flow_zonecodes)
        with open(outputs["ArrowsFlowsPrivate2045"], 'w') as f:
            dump(flow_pr, f)

        logger.warning("JtW model" + str(Scenario) + " cbar [public, private] = " + str(cbar_k))

        # Calculate predicted population
        DjPred = np.zeros(n)
        for k in range(len(Tij)):
            DjPred += Tij[k].sum(axis=1)
        # Create a dataframe with Zone and people count
        DjPred = pd.DataFrame(DjPred, columns=['population'])
        DjPred['zone'] = zonecodes_ATH_list

        end = time.perf_counter()
        # print("Journey to work model run elapsed time (secs)=", end - start)
        logger.warning("Journey to work model run elapsed time (mins)=" + str((end - start) / 60))

        return DjPred

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



