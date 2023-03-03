# The keys of the inputs and outputs dictionaries, as well as the file names should follow the camelCase notation.

inputs = {}

inputs["ZoneCodesFile"]          = "./HARMONY_LUTI_ATH/external-data/ATH_ZoneCodes.csv"
inputs["ZoneCoordinates"]        = "./HARMONY_LUTI_ATH/external-data/zones_data_coordinates_ath.csv"
inputs["IntrazoneDist"]          = "./HARMONY_LUTI_ATH/external-data/intrazone_distances.csv"

inputs["CijPrivateMinFilename"]  = "./HARMONY_LUTI_ATH/external-data/Cij_private.csv"
inputs["CijPublicMinFilename"]   = "./HARMONY_LUTI_ATH/external-data/Cij_public.csv"

inputs["DataEmployment2019"]     = "./HARMONY_LUTI_ATH/external-data/Employment/employment_ath.csv"
inputs["DataEmployment2030"]     = "./HARMONY_LUTI_ATH/external-data/Employment/employment_ath_2030.csv"
inputs["DataEmployment2045"]     = "./HARMONY_LUTI_ATH/external-data/Employment/employment_ath_2045.csv"

inputs["HhFloorspace2019"]       = "./HARMONY_LUTI_ATH/external-data/Employment/hh_floorspace.csv"
inputs["HhFloorspace2045"]       = "./HARMONY_LUTI_ATH/external-data/Employment/hh_floorspace_2045.csv"

inputs["DataZonesShapefile"]     = "./HARMONY_LUTI_ATH/external-data/geography/Athens_zones.shp"
inputs["ZoneCentroidsShapefile"] = "./HARMONY_LUTI_ATH/external-data/geography/Athens_zone_centroids.shp"
inputs["ZoneCentroidsShapefileWGS84"] = "./HARMONY_LUTI_ATH/external-data/geography/Athens_zone_centroids_WGS84.shp"
inputs["RoadNetworkShapefile"]   = "./HARMONY_LUTI_TUR/external-data/geography/Main_roads.shp"

outputs = {}

outputs["JobsAccessibility2019"]               = "./Outputs-Athens/Jobs_accessibility_2019.csv"
outputs["JobsAccessibility2030"]               = "./Outputs-Athens/Jobs_accessibility_2030.csv"
outputs["JobsAccessibility2045"]               = "./Outputs-Athens/Jobs_accessibility_2045.csv"
outputs["HousingAccessibility2019"]            = "./Outputs-Athens/Housing_accessibility_2019.csv"
outputs["HousingAccessibility2030"]            = "./Outputs-Athens/Housing_accessibility_2030.csv"
outputs["HousingAccessibility2045"]            = "./Outputs-Athens/Housing_accessibility_2045.csv"
outputs["EjOi2019"]                            = "./Outputs-Athens/EiDjOi_2019.csv"
outputs["EjOi2030"]                            = "./Outputs-Athens/EiDjOi_2030.csv"
outputs["EjOi2045"]                            = "./Outputs-Athens/EiDjOi_2045.csv"
outputs["JobsProbTijPublic2019"]               = "./Outputs-Athens/jobs_probTij_public_2019.csv"
outputs["JobsProbTijPrivate2019"]              = "./Outputs-Athens/jobs_probTij_private_2019.csv"
outputs["JobsProbTijPublic2030"]               = "./Outputs-Athens/jobs_probTij_public_2030.csv"
outputs["JobsProbTijPrivate2030"]              = "./Outputs-Athens/jobs_probTij_private_2030.csv"
outputs["JobsProbTijPublic2045"]               = "./Outputs-Athens/jobs_probTij_public_2045.csv"
outputs["JobsProbTijPrivate2045"]              = "./Outputs-Athens/jobs_probTij_private_2045.csv"
outputs["JobsTijPublic2019"]                   = "./Outputs-Athens/jobs_Tij_public_2019.csv"
outputs["JobsTijPrivate2019"]                  = "./Outputs-Athens/jobs_Tij_private_2019.csv"
outputs["JobsTijPublic2030"]                   = "./Outputs-Athens/jobs_Tij_public_2030.csv"
outputs["JobsTijPrivate2030"]                  = "./Outputs-Athens/jobs_Tij_private_2030.csv"
outputs["JobsTijPublic2045"]                   = "./Outputs-Athens/jobs_Tij_public_2045.csv"
outputs["JobsTijPrivate2045"]                  = "./Outputs-Athens/jobs_Tij_private_2045.csv"
outputs["ArrowsFlowsPublic2019"]               = "./Outputs-Athens/flows_2019_pu.geojson"
outputs["ArrowsFlowsPrivate2019"]              = "./Outputs-Athens/flows_2019_pr.geojson"
outputs["ArrowsFlowsPublic2030"]               = "./Outputs-Athens/flows_2030_pu.geojson"
outputs["ArrowsFlowsPrivate2030"]              = "./Outputs-Athens/flows_2030_pr.geojson"
outputs["ArrowsFlowsPublic2045"]               = "./Outputs-Athens/flows_2045_pu.geojson"
outputs["ArrowsFlowsPrivate2045"]              = "./Outputs-Athens/flows_2045_pr.geojson"
outputs["MapPopChange20192030"]                = "./Outputs-Athens/pop_change_19-30.png"
outputs["MapPopChange20302045"]                = "./Outputs-Athens/pop_change_30-45.png"
outputs["MapPopChange20192045"]                = "./Outputs-Athens/pop_change_19-45.png"
outputs["MapHousingAccChange20192030Public"]   = "./Outputs-Athens/HA_change_19-30_pu.png"
outputs["MapHousingAccChange20192030Private"]  = "./Outputs-Athens/HA_change_19-30_pr.png"
outputs["MapHousingAccChange20192045Public"]   = "./Outputs-Athens/HA_change_19-45_pu.png"
outputs["MapHousingAccChange20192045Private"]  = "./Outputs-Athens/HA_change_19-45_pr.png"
outputs["MapHousingAccChange20302045Public"]   = "./Outputs-Athens/HA_change_30-45_pu.png"
outputs["MapHousingAccChange201302045Private"] = "./Outputs-Athens/HA_change_30-45_pr.png"
outputs["MapJobsAccChange20192030Public"]      = "./Outputs-Athens/JA_change_19-30_pu.png"
outputs["MapJobsAccChange20192030Private"]     = "./Outputs-Athens/JA_change_19-30_pr.png"
outputs["MapJobsAccChange20192045Public"]      = "./Outputs-Athens/JA_change_19-45_pu.png"
outputs["MapJobsAccChange20192045Private"]     = "./Outputs-Athens/JA_change_19-45_pr.png"
outputs["MapJobsAccChange20302045Public"]      = "./Outputs-Athens/JA_change_30-45_pu.png"
outputs["MapJobsAccChange20302045Private"]     = "./Outputs-Athens/JA_change_30-45_pr.png"
outputs["MapResultsShapefile"]                 = "./Outputs-Athens/ATH_results.shp"
