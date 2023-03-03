"""
costs.py
Create costs matrices e.g. from an MSOA->MSOA matrix, make an MSOA to geocoded point cost matrix
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points


from HARMONY_LUTI_OXF.globals import modelRunsDir, QUANTCijRoadCentroidsFilename, QUANTCijBusCentroidsFilename, QUANTCijRailCentroidsFilename

"""
costMSOAToPoint
Take the MSOA to MSOA cost matrix and build a variant for MSOA to a point dataset based on 
MSOA to MSOA plus an additional straight line distance from the point to the nearest
MSOA centroid

The Plan:
    for each point in points
    find closest msoa
    find distance offset to closest msoa and delta time (crowfly time to this nearest msoa to point)
    foreach MSOA
    lookup MSOA to closest MSOA (from cij) plus delta time
    write MSOA->point matrix with cij + delta time to point

NOTE: also see databuilder.computeGeolytixCosts() for the same basic methodology
PRE: uses the QUANTCijRoadCentroidsFilename file from model-runs for the cij centroid points - not same as zonecodes
@param cij cost matrix for MSOA to MSOA, zones identified by zonecodes (numpy array)
@param dfPoints the DataFrame containing [zonei,east,north] for our points
@returns an matrix that is count(points) x count(zonecodes) (i.e. cols x rows so [pointi,zonecodei])
"""

def costMSOAToPoint_3modes(cij_roads, cij_bus, cij_rail, dfPoints, OXF_MSOA_list):
    # const to define what speed we travel the additional distance to the retail point e.g. 30mph = 13 ms-1
    metresPerSec_roads = 13.0 # 30 miles/h = 47 km/h
    metresPerSec_bus = 13.0  # 30 miles/h = 47 km/h
    metresPerSec_rail = 13.0  # 30 miles/h = 47 km/h

    # read in the road, bus and rail centroids for the cij matrix
    df_roads = pd.read_csv(os.path.join(modelRunsDir, QUANTCijRoadCentroidsFilename), index_col='zonecode')
    df_bus = pd.read_csv(os.path.join(modelRunsDir, QUANTCijBusCentroidsFilename), index_col='zonecode')
    df_rail = pd.read_csv(os.path.join(modelRunsDir, QUANTCijRailCentroidsFilename), index_col='zonecode')

    # Filter out Oxfordshire from EWS files
    df_roads = df_roads.loc[OXF_MSOA_list]  # Filter rows
    df_bus = df_bus.loc[OXF_MSOA_list]  # Filter rows
    df_rail = df_rail.loc[OXF_MSOA_list]  # Filter rows

    df_roads['zonecode'] = df_roads.index  # turn the index (i.e. MSOA codes) back into a columm
    df_bus['zonecode'] = df_bus.index  # turn the index (i.e. MSOA codes) back into a columm
    df_rail['zonecode'] = df_rail.index  # turn the index (i.e. MSOA codes) back into a columm

    # Reset indexes
    df_roads.reset_index(drop=True, inplace=True)
    df_bus.reset_index(drop=True, inplace=True)
    df_rail.reset_index(drop=True, inplace=True)

    # Overwrite the zonei column with the new index
    df_roads['zonei'] = df_roads.index
    df_bus['zonei'] = df_bus.index
    df_rail['zonei'] = df_rail.index

    # code this into a geodataframe so we can make a spatial index
    gdf_roads = gpd.GeoDataFrame(df_roads, crs='epsg:4326', geometry=gpd.points_from_xy(df_roads.vertex_lon, df_roads.vertex_lat))
    gdf_bus = gpd.GeoDataFrame(df_bus, crs='epsg:4326', geometry=gpd.points_from_xy(df_bus.vertex_lon, df_bus.vertex_lat))
    gdf_rail = gpd.GeoDataFrame(df_rail, crs='epsg:4326', geometry=gpd.points_from_xy(df_rail.vertex_lon, df_rail.vertex_lat))

    # but it's lat/lon and we want east/north, convert crs:
    centroids_roads = gdf_roads.to_crs("EPSG:27700")
    centroids_bus = gdf_bus.to_crs("EPSG:27700")
    centroids_rail = gdf_rail.to_crs("EPSG:27700")

    dest_unary_roads = centroids_roads["geometry"].unary_union  # and need this join for the centroid points nearest lookup
    dest_unary_bus = centroids_bus["geometry"].unary_union  # and need this join for the centroid points nearest lookup
    dest_unary_rail = centroids_rail["geometry"].unary_union  # and need this join for the centroid points nearest lookup

    # create a new MSOA to points cost matix
    m, n = cij_roads.shape
    p, cols = dfPoints.shape
    # print("array size = ", p, m)

    cijpoint_roads = np.zeros(m * p, dtype=np.float).reshape(m, p)  # so m=MSOA and p=points index
    cijpoint_bus = np.zeros(m * p, dtype=np.float).reshape(m, p)  # so m=MSOA and p=points index
    cijpoint_rail = np.zeros(m * p, dtype=np.float).reshape(m, p)  # so m=MSOA and p=points index

    # now make the amended cost function
    count = 0
    for row in dfPoints.itertuples(index=False):  # NOTE: iterating over Pandas rows is supposed to be bad - how else to do this?
        if (count % 50 == 0): print("costs::costMSOAToPoint ", count, "/", p)
        count += 1

        p_zonei = getattr(row, 'zonei')
        p_east = getattr(row, 'east')
        p_north = getattr(row, 'north')

        near_roads = nearest_points(Point(p_east, p_north), dest_unary_roads)
        near_bus = nearest_points(Point(p_east, p_north), dest_unary_bus)
        near_rail = nearest_points(Point(p_east, p_north), dest_unary_rail)

        match_geom_roads = centroids_roads.loc[centroids_roads.geometry == near_roads[1]]
        match_geom_bus = centroids_bus.loc[centroids_bus.geometry == near_bus[1]]
        match_geom_rail = centroids_rail.loc[centroids_rail.geometry == near_rail[1]]

        pmsoa_zonei_roads = int(match_geom_roads.zonei)  # closest point msoa zone
        pmsoa_zonei_bus = int(match_geom_bus.zonei)  # closest point msoa zone
        pmsoa_zonei_rail = int(match_geom_rail.zonei)  # closest point msoa zone

        pmsoa_pt_roads = match_geom_roads.geometry
        pmsoa_pt_bus = match_geom_bus.geometry
        pmsoa_pt_rail = match_geom_rail.geometry

        pmsoa_east_roads = float(pmsoa_pt_roads.centroid.x)
        pmsoa_east_bus = float(pmsoa_pt_bus.centroid.x)
        pmsoa_east_rail = float(pmsoa_pt_rail.centroid.x)

        pmsoa_north_roads = float(pmsoa_pt_roads.centroid.y)
        pmsoa_north_bus = float(pmsoa_pt_bus.centroid.y)
        pmsoa_north_rail = float(pmsoa_pt_rail.centroid.y)

        dx_roads = p_east - pmsoa_east_roads
        dx_bus = p_east - pmsoa_east_bus
        dx_rail = p_east - pmsoa_east_rail

        dy_roads = p_north - pmsoa_north_roads
        dy_bus = p_north - pmsoa_north_bus
        dy_rail = p_north - pmsoa_north_rail

        dist_roads = np.sqrt(dx_roads * dx_roads + dy_roads * dy_roads)  # dist between point and centroid used for shortest path
        dist_bus = np.sqrt(dx_bus * dx_bus + dy_bus * dy_bus)  # dist between point and centroid used for shortest path
        dist_rail = np.sqrt(dx_rail * dx_rail + dy_rail * dy_rail)  # dist between point and centroid used for shortest path

        # work out an additional delta cost based on increased time getting from this point to the centroid
        deltaCost_roads = (dist_roads / metresPerSec_roads) / 60.0  # transit time in mins
        deltaCost_bus = (dist_bus / metresPerSec_bus) / 60.0  # transit time in mins
        deltaCost_rail = (dist_rail / metresPerSec_rail) / 60.0  # transit time in mins

        # now write every cij value for msoa_zonei to p_zonei (closest) PLUS deltaCose for p_zonei to actual point
        for i in range(n):
            C1_roads = cij_roads[pmsoa_zonei_roads, i]  # yes, this is right for a trip from MSOA to closest point MSOA - QUANT is BACKWARDS
            C1_bus = cij_bus[pmsoa_zonei_bus, i]  # yes, this is right for a trip from MSOA to closest point MSOA - QUANT is BACKWARDS
            C1_rail = cij_rail[pmsoa_zonei_rail, i]  # yes, this is right for a trip from MSOA to closest point MSOA - QUANT is BACKWARDS

            cijpoint_roads[i, p_zonei] = C1_roads + deltaCost_roads
            cijpoint_bus[i, p_zonei] = C1_bus + deltaCost_bus
            cijpoint_rail[i, p_zonei] = C1_rail + deltaCost_rail

            # NOTE: you can only go in one direction with a matrix that is asymmetric

    return cijpoint_roads, cijpoint_bus, cijpoint_rail

################################################################################