"""
globals.py

Globals used by model
"""
import os

# Directories paths
modelRunsDir = "./HARMONY_LUTI_ATH/model-runs"

# Zones and attractors
data_HH_zones = os.path.join(modelRunsDir,"jobs_Pop_Zones.csv")
data_HH_attractors = os.path.join(modelRunsDir,"jobs_HH_Attractors.csv")

# Merged Csvs
Pop_Change = os.path.join(modelRunsDir, "Population_change.csv")
HA_Change = os.path.join(modelRunsDir, "Housing_Accessibility_change.csv")
Job_Change = os.path.join(modelRunsDir, "Jobs_Accessibility_change.csv")