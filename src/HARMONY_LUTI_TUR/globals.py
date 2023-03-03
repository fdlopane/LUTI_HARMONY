"""
globals.py

Globals used by model
"""
import os

# Directories paths
modelRunsDir = "./HARMONY_LUTI_TUR/model-runs"

### Journey to work model

# Merged Csvs
Pop_Change = os.path.join(modelRunsDir, "Population_change.csv")
HA_Change = os.path.join(modelRunsDir, "Housing_Accessibility_change.csv")
Job_Change = os.path.join(modelRunsDir, "Jobs_Accessibility_change.csv")

### Schools model
# Primary schools zones and attractors
data_primary_zones = os.path.join(modelRunsDir,"primaryZones.csv")
data_primary_attractors = os.path.join(modelRunsDir,"primaryAttractors_2019.csv") #number of primary school pupils

# Middle schools zones and attractors
data_middle_zones = os.path.join(modelRunsDir,"middleZones_2019.csv")
data_middle_attractors = os.path.join(modelRunsDir,"middleAttractors_2019.csv") #number of middle school pupils

# High schools zones and attractors
data_high_zones = os.path.join(modelRunsDir,"highZones_2019.csv")
data_high_attractors = os.path.join(modelRunsDir,"highAttractors_2019.csv") #number of high school pupils

# Universities schools zones and attractors
data_unis_zones = os.path.join(modelRunsDir,"unisZones_2019.csv")
data_unis_zones_2030 = os.path.join(modelRunsDir,"unisZones_2030.csv")
data_unis_attractors = os.path.join(modelRunsDir,"unisAttractors_2019.csv") #number of students
data_unis_attractors_2030 = os.path.join(modelRunsDir,"unisAttractors_2030.csv")

### Hospitals model

# Zones and attractors
data_hospital_zones = os.path.join(modelRunsDir,"hospitalZones.csv")
data_hospital_attractors = os.path.join(modelRunsDir,"hospitalAttractors.csv") #Number of beds
data_hospital_zones_2030 = os.path.join(modelRunsDir,"hospitalZones2030.csv")
data_hospital_attractors_2030 = os.path.join(modelRunsDir,"hospitalAttractors2030.csv") #Number of beds