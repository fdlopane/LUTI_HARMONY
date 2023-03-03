"""
globals.py

Globals used by model
"""
import os

########################################################################################################################
# Directories paths
modelRunsDir = "./HARMONY_LUTI_OXF/model-runs"

########################################################################################################################
# These are download urls for big external data that can't go in the GitHub repo
url_QUANTCijRoadMinFilename = "https://liveuclac-my.sharepoint.com/:u:/g/personal/ucfnrmi_ucl_ac_uk/EZd4HZVVHd1OuZ_Qj3uKGNcBSe_OoG6unjrVbAyRvGquaQ?e=LxuwMv&download=1"
url_QUANT_ZoneCodes = "https://liveuclac-my.sharepoint.com/:x:/g/personal/ucfnrmi_ucl_ac_uk/EdlPQ9GtHsFBigZ_sUnOKX0BqJB38g_TeqX8NorvojelfQ?e=6ZsPBE&download=1"
url_QUANT_RoadCentroids = "https://osf.io/usm84/download"
url_QUANT_BusCentroids = "https://osf.io/k4xwv/download"
url_QUANT_RailCentroids = "https://osf.io/ycaqb/download"

########################################################################################################################
# File names (no complete path as they might be present in more folders with the same name)
# e.g. check that this file is in AAA folder, otherwise load it from BBB folder
ZoneCodesFilename = 'EWS_ZoneCodes.csv'

#cost matrix names
QUANTCijRoadMinFilename = 'dis_roads_min.bin'
QUANTCijBusMinFilename = 'dis_bus_min.bin'
QUANTCijRailMinFilename = 'dis_gbrail_min.bin'

QUANTCijRoadMinFilename_OXF = 'Cij_road_min_OXF.bin' # England and Wales
QUANTCijBusMinFilename_OXF = 'Cij_bus_min_OXF.bin' # England and Wales
QUANTCijRailMinFilename_OXF = 'Cij_gbrail_min_OXF.bin' # England and Wales

CijRoadMinFilename = 'Cij_road_min.bin'
CijBusMinFilename = 'Cij_bus_min.bin'
CijRailMinFilename = 'Cij_gbrail_min.bin'

SObsRoadFilename_OXF = 'SObs_1_OXF.bin'
SObsBusFilename_OXF = 'SObs_2_OXF.bin'
SObsRailFilename_OXF = 'SObs_3_OXF.bin'

#centroids for the cost matrices
QUANTCijRoadCentroidsFilename = 'roadcentroidlookup_QC.csv'
QUANTCijBusCentroidsFilename = 'buscentroidlookup_QC.csv'
QUANTCijRailCentroidsFilename = 'gbrailcentroidlookup_QC.csv'

########################################################################################################################
# -- INPUT FILES --

# Retail data
data_open_geolytix_regression_OXF = os.path.join(modelRunsDir,"geolytix_retailpoints_open_regression_OXF.csv")

# Schools data
data_schools_england_primary = os.path.join(modelRunsDir,"OSF_RAMPUrbanAnalytics/primary_england.csv")
data_schools_OXF_primary = os.path.join(modelRunsDir,"primary_OXF.csv")
data_schools_england_secondary = os.path.join(modelRunsDir,"OSF_RAMPUrbanAnalytics/secondary_england.csv")
data_schools_OXF_secondary = os.path.join(modelRunsDir,"secondary_OXF.csv")

# Census data
data_census_QS103_OXF = os.path.join(modelRunsDir,"QS103EW_MSOA_OXF.csv")

# Hospitals
data_hospitals_OXF = os.path.join(modelRunsDir, "NHS_join_mod_OXF.csv") #data on hospitals in Oxfordshire

# Employment
HH_floorspace_OXF = os.path.join(modelRunsDir,"FS_OA1.0_OXF.csv")

########################################################################################################################
# -- OUTPUT FILES --

### Journey to work model
# Employment
data_jobs_employment = os.path.join(modelRunsDir,"jobsEmployment.csv")
Jobs_DjOi_2011 = os.path.join(modelRunsDir, "Jobs_DjOi_2011.csv")

# Zones and attractors
data_HH_zones_2011 = os.path.join(modelRunsDir,"jobs_Pop_Zones_2011.csv")
data_HH_zones_2019 = os.path.join(modelRunsDir,"jobs_Pop_Zones_2019.csv")
data_HH_zones_2030 = os.path.join(modelRunsDir,"jobs_Pop_Zones_2030.csv")

data_HH_attractors_2011 = os.path.join(modelRunsDir,"jobs_HH_Attractors_2011.csv")
data_HH_attractors_2019 = os.path.join(modelRunsDir,"jobs_HH_Attractors_2019.csv")
data_HH_attractors_2030 = os.path.join(modelRunsDir,"jobs_HH_Attractors_2030.csv")

# Probabilities
data_jobs_probTij_roads_2011_csv = os.path.join(modelRunsDir,"jobsProbTij_roads_2011.csv")
data_jobs_probTij_bus_2011_csv = os.path.join(modelRunsDir,"jobsProbTij_bus_2011.csv")
data_jobs_probTij_rail_2011_csv = os.path.join(modelRunsDir,"jobsProbTij_rail_2011.csv")

# Flows
data_jobs_Tij_roads_2011_csv = os.path.join(modelRunsDir,"jobsTij_roads_2011.csv")
data_jobs_Tij_bus_2011_csv = os.path.join(modelRunsDir,"jobsTij_bus_2011.csv")
data_jobs_Tij_rail_2011_csv = os.path.join(modelRunsDir,"jobsTij_rail_2011.csv")

# Job accessibility
data_jobs_accessibility_2011 = os.path.join(modelRunsDir,"jobs_accessibility_2011.csv")

# Housing accessibility
data_housing_accessibility_2011 = os.path.join(modelRunsDir,"housing_accessibility_2011.csv")

# Merged Csvs
Pop_Change = os.path.join(modelRunsDir, "Population_change.csv")
HA_Change = os.path.join(modelRunsDir, "Housing_Accessibility_change.csv")
Job_Change = os.path.join(modelRunsDir, "Jobs_Accessibility_change.csv")

### Retail (Income) model
# Cost matrices
data_retailpoints_cij_roads = os.path.join(modelRunsDir,"retailpointsCij_roads.bin")
data_retailpoints_cij_bus = os.path.join(modelRunsDir,"retailpointsCij_bus.bin")
data_retailpoints_cij_rail = os.path.join(modelRunsDir,"retailpointsCij_rail.bin")

data_retailpoints_cij_roads_csv = os.path.join(modelRunsDir,"retailpointsCij_roads.csv")
data_retailpoints_cij_bus_csv = os.path.join(modelRunsDir,"retailpointsCij_bus.csv")
data_retailpoints_cij_rail_csv = os.path.join(modelRunsDir,"retailpointsCij_rail.csv")
# Zones and attractors
data_retailpoints_zones = os.path.join(modelRunsDir,"retailpointsZones.csv")
data_retailpoints_attractors = os.path.join(modelRunsDir,"retailpointsAttractors.csv")
# Population
data_retailpoints_population_EWS = os.path.join(modelRunsDir,"retailpointsPopulation_EWS.csv")
data_retailpoints_population_EW = os.path.join(modelRunsDir,"retailpointsPopulation_EW.csv")
data_retailpoints_population_OXF = os.path.join(modelRunsDir,"retailpointsPopulation_OXF.csv")
# Probabilities
data_retailpoints_probRij_roads = os.path.join(modelRunsDir,"retailpointsProbRij_roads.bin")
data_retailpoints_probRij_bus = os.path.join(modelRunsDir,"retailpointsProbRij_bus.bin")
data_retailpoints_probRij_rail = os.path.join(modelRunsDir,"retailpointsProbRij_rail.bin")
# Flows
data_retailpoints_Rij_roads = os.path.join(modelRunsDir,"retailpointsRij_roads.bin")
data_retailpoints_Rij_bus = os.path.join(modelRunsDir,"retailpointsRij_bus.bin")
data_retailpoints_Rij_rail = os.path.join(modelRunsDir,"retailpointsRij_rail.bin")

### Retail (Population) model
# Zones and attractors
data_populationretail_zones = os.path.join(modelRunsDir,"populationretailZones.csv")
data_populationretail_attractors = os.path.join(modelRunsDir,"populationretailAttractors.csv")
# Population
data_populationretail_population_OXF_2011 = os.path.join(modelRunsDir,"populationretailPopulation_OXF_2011.csv")
data_populationretail_population_OXF_2019 = os.path.join(modelRunsDir,"populationretailPopulation_OXF_2019.csv")
data_populationretail_population_OXF_2030 = os.path.join(modelRunsDir,"populationretailPopulation_OXF_2030.csv")
# Probabilities
data_populationretail_probRij_roads_2011_csv = os.path.join(modelRunsDir,"populationretailProbRij_roads_2011.csv")
data_populationretail_probRij_bus_2011_csv = os.path.join(modelRunsDir,"populationretailProbRij_bus_2011.csv")
data_populationretail_probRij_rail_2011_csv = os.path.join(modelRunsDir,"populationretailProbRij_rail_2011.csv")
# Flows
data_populationretail_Rij_roads_2011_csv = os.path.join(modelRunsDir,"populationretailRij_roads_2011.csv")
data_populationretail_Rij_bus_2011_csv = os.path.join(modelRunsDir,"populationretailRij_bus_2011.csv")
data_populationretail_Rij_rail_2011_csv = os.path.join(modelRunsDir,"populationretailRij_rail_2011.csv")

### Schools model
# Primary schools cost matrices
data_primary_cij_roads = os.path.join(modelRunsDir,"primaryCij_roads.bin")
data_primary_cij_bus = os.path.join(modelRunsDir,"primaryCij_bus.bin")
data_primary_cij_rail = os.path.join(modelRunsDir,"primaryCij_rail.bin")
data_primary_cij_roads_csv = os.path.join(modelRunsDir,"primaryCij_roads.csv")
data_primary_cij_bus_csv = os.path.join(modelRunsDir,"primaryCij_bus.csv")
data_primary_cij_rail_csv = os.path.join(modelRunsDir,"primaryCij_rail.csv")
# Primary schools zones and attractors
data_primary_zones = os.path.join(modelRunsDir,"primaryZones.csv")
data_primary_attractors = os.path.join(modelRunsDir,"primaryAttractors.csv")
# Primary schools population
data_schoolagepopulation_englandwales = os.path.join(modelRunsDir,"schoolagepopulation_englandwales_msoa.csv")
data_schoolagepopulation_scotland = os.path.join(modelRunsDir,"schoolagepopulation_scotland_iz.csv")
data_schoolagepopulation = os.path.join(modelRunsDir,"schoolagepopulation_englandwalesscotland_msoaiz.csv")
data_primary_population_2011 = os.path.join(modelRunsDir,"primaryPopulation_2011.csv")
data_primary_population_2019 = os.path.join(modelRunsDir,"primaryPopulation_2019.csv")
data_primary_population_2030 = os.path.join(modelRunsDir,"primaryPopulation_2030.csv")
# Primary schools probabilities
data_primary_probPij_roads_2011_csv = os.path.join(modelRunsDir,"primaryProbPij_roads_2011.csv")
data_primary_probPij_bus_2011_csv = os.path.join(modelRunsDir,"primaryProbPij_bus_2011.csv")
data_primary_probPij_rail_2011_csv = os.path.join(modelRunsDir,"primaryProbPij_rail_2011.csv")
# Secondary schools cost matrices
data_secondary_cij_roads = os.path.join(modelRunsDir,"secondaryCij_roads.bin")
data_secondary_cij_bus = os.path.join(modelRunsDir,"secondaryCij_bus.bin")
data_secondary_cij_rail = os.path.join(modelRunsDir,"secondaryCij_rail.bin")
data_secondary_cij_roads_csv = os.path.join(modelRunsDir,"secondaryCij_roads.csv")
data_secondary_cij_bus_csv = os.path.join(modelRunsDir,"secondaryCij_bus.csv")
data_secondary_cij_rail_csv = os.path.join(modelRunsDir,"secondaryCij_rail.csv")
# Secondary school zones and attractors
data_secondary_zones = os.path.join(modelRunsDir,"secondaryZones.csv")
data_secondary_attractors = os.path.join(modelRunsDir,"secondaryAttractors.csv")
# Secondary schools population
data_secondary_population_2011 = os.path.join(modelRunsDir,"secondaryPopulation_2011.csv")
data_secondary_population_2019 = os.path.join(modelRunsDir,"secondaryPopulation_2019.csv")
data_secondary_population_2030 = os.path.join(modelRunsDir,"secondaryPopulation_2030.csv")
# Secondary schools probabilities
data_secondary_probPij_roads_2011_csv = os.path.join(modelRunsDir,"secondaryProbPij_roads_2011.csv")
data_secondary_probPij_bus_2011_csv = os.path.join(modelRunsDir,"secondaryProbPij_bus_2011.csv")
data_secondary_probPij_rail_2011_csv = os.path.join(modelRunsDir,"secondaryProbPij_rail_2011.csv")
# Primary schools flows:
data_primary_Pij_roads_2011_csv = os.path.join(modelRunsDir,"primaryPij_roads_2011.csv")
data_primary_Pij_bus_2011_csv = os.path.join(modelRunsDir,"primaryPij_bus_2011.csv")
data_primary_Pij_rail_2011_csv = os.path.join(modelRunsDir,"primaryPij_rail_2011.csv")
# Secondary schools flows
data_secondary_Pij_roads_2011_csv = os.path.join(modelRunsDir,"secondaryPij_roads_2011.csv")
data_secondary_Pij_bus_2011_csv = os.path.join(modelRunsDir,"secondaryPij_bus_2011.csv")
data_secondary_Pij_rail_2011_csv = os.path.join(modelRunsDir,"secondaryPij_rail_2011.csv")

### Hospitals model
# Cost matrices
data_hospital_cij_roads = os.path.join(modelRunsDir,"hospitalCij_roads.bin")
data_hospital_cij_bus = os.path.join(modelRunsDir,"hospitalCij_bus.bin")
data_hospital_cij_rail = os.path.join(modelRunsDir,"hospitalCij_rail.bin")
data_hospital_cij_roads_csv = os.path.join(modelRunsDir,"hospitalCij_roads.csv")
data_hospital_cij_bus_csv = os.path.join(modelRunsDir,"hospitalCij_bus.csv")
data_hospital_cij_rail_csv = os.path.join(modelRunsDir,"hospitalCij_rail.csv")
# Zones and attractors
data_hospital_zones = os.path.join(modelRunsDir,"hospitalZones.csv")
data_hospital_attractors = os.path.join(modelRunsDir,"hospitalAttractors.csv")
# Population
data_totalpopulation = os.path.join(modelRunsDir,"totalpopulation_englandwalesscotland_msoaiz.csv") #this is QS103 col All People joined for E+W+S
data_agepopulation = os.path.join(modelRunsDir,"agepopulation_englandwalesscotland_msoaiz.csv") #same as total pop, but with age breakdown
data_hospital_population = os.path.join(modelRunsDir,"hospitalPopulation.csv")
# Probabilities
data_hospital_probHij_roads_2011_csv = os.path.join(modelRunsDir,"hospitalProbHij_roads_2011.csv")
data_hospital_probHij_bus_2011_csv = os.path.join(modelRunsDir,"hospitalProbHij_bus_2011.csv")
data_hospital_probHij_rail_2011_csv = os.path.join(modelRunsDir,"hospitalProbHij_rail_2011.csv")
# Flows 2011
data_hospital_Hij_roads_2011_csv = os.path.join(modelRunsDir,"hospitalHij_roads_2011.csv")
data_hospital_Hij_bus_2011_csv = os.path.join(modelRunsDir,"hospitalHij_bus_2011.csv")
data_hospital_Hij_rail_2011_csv = os.path.join(modelRunsDir,"hospitalHij_rail_2011.csv")
########################################################################################################################
