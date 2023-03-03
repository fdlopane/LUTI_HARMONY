"""
DEPRECATED - until this does anything more thn just load the data into a map, I'm using Panda.read_csv instead

zonecodes.py
Handle creation, loading and operations on the zone codes lookup table.
This is the table that contains the mapping between the zone number and the zone area key code (i.e. MSOA). It also
stores additional information like the lat/lon of the zone centroid.
"""

import os.path
import geopandas as gpd
import csv
from pyproj import Proj, transform

from HARMONY_LUTI_OXF.globals import *
#from utils import loadZoneLookup

################################################################################
"""
Data class to represent a row of the zone codes lookup table
"""
# class ZoneData:
#     def __init__(self):
#         self.areakey = ""
#         self.zonei = -1
#         self.name = ""
#         self.lat = 0
#         self.lon = 0
#         self.osgb36East = 0
#         self.osgb36North = 0
#         self.area = 0

################################################################################


class ZoneCodes:

    def __init__(self):
        self.dt = {} #dict of PK=areakey string, containing ZoneData
        #todo: need load here!

    ################################################################################

    @staticmethod
    def fromFile():
        #todo: really want this to be a static singleton...
        zc = ZoneCodes()
        zc.dt = ZoneCodes.loadZoneLookup(os.path.join(modelRunsDir,ZoneCodesFilename))
        return zc

    ################################################################################

    """
    Load the zone codes lookup from a csv file into a dictionary of dictionaries
    """
    @staticmethod
    def loadZoneLookup(filename):
        ZoneLookup = {}
        with open(filename,'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            header = next(reader,None) #skip header line
            for row in reader:
                #zonei,areakey,lat,lon,osgb36_east,osgb36_north,area
                #0,E02000001,51.515,-0.09051585,532482.75,181269.56,2897837.5
                zonei = int(row[0])
                msoa = row[1]
                lat = float(row[2])
                lon = float(row[3])
                east = float(row[4])
                north = float(row[5])
                area = float(row[6])
                ZoneLookup[msoa] = {
                    'areakey': msoa,
                    'zonei': zonei,
                    'east': east,
                    'north': north,
                    'lon': lon,
                    'lat': lat,
                    'area': area
                }
                #print("loadZoneLookup: ",msoa,zonei,name,east,north) #DEBUG
        return ZoneLookup

    ###############################################################################


    # /// <summary>
    # /// Everything relies on a table containing the area key to zone code index lookup.
    # /// This is calculated from the shapefile, by going though all the areas and numbering them (from zero).
    # /// The number of areas found determines the order of the T and dis matrices.
    # /// There is a potential problem here if the trips file contains an area not in the shapefile. If we don't know the
    # /// location of an area then we have to forget about it.
    # /// This is static as all the other methods require this to be created first.
    # /// </summary>
    # /// <param name="Shpfilename">In OSGB36</param>
    # /// <returns>A data table containing the areakey to zone code lookup</returns>
    # @staticmethod
    # def deriveAreakeyToZoneCodeFromShapefile(ShpFilenameOSGB36):
    #     #read shapefile and create areakey, zonei, centroid and area table

    #     dt = {} #primary key is areakey

    #     #go through shapefile and write unique zones to the table, along with their (lat/lon) centroids and areas
    #     UniqueAreaKeys = []
    #     #load the shapefile
    #     features = gpd.read_file(ShpFilenameOSGB36)
    #     #now process the shapefile
    #     inProj = Proj(features.crs)
    #     outProj = Proj(init='epsg:4326')

    #     for idx, f in features.iterrows():
    #         geomOSGB36 = f['geometry']
    #         centroidwgs84x, centroidwgs84y = transform(inProj, outProj, geomOSGB36.centroid.x, geomOSGB36.centroid.y)
    #         #now need the area key code
    #         #there are only three columns in the shapefile, the_geom (=0), code (=1, MSOA11CA) and plain text name (=2, MSOA11NM)
    #         AreaKey = f["MSOA11CD"]
    #         Name = f["MSOA11NM"]
    #         if not AreaKey in UniqueAreaKeys:
    #             dt[AreaKey] = {
    #                 'areakey': AreaKey,
    #                 'zonei': len(UniqueAreaKeys),
    #                 'name': Name,
    #                 'east': geomOSGB36.centroid.x,
    #                 'north': geomOSGB36.centroid.y,
    #                 'lat': centroidwgs84y,
    #                 'lon': centroidwgs84x
    #             }
    #             UniqueAreaKeys.append(AreaKey)
    #         else:
    #             #else warn duplicate area? might happen if islands split, then have to be careful about centroids and areas
    #             print("WARNING: Duplicate area: " + AreaKey)
    #         #endif
    #     #end for

    #     #save data table - as csv data! NOTE: this doesn't come out in zonei order
    #     with open(os.path.join(modelRunsDir,ZoneCodesFilename),"w") as file:
    #         file.write("zonei,areakey,name,east,north,lon,lat\n")
    #         sorted_dt = sorted(dt.values(), key=lambda k: k['zonei']) #in Python we have to sort the dict values for them to come out in zonei order in the csv file
    #         for row in sorted_dt:
    #             file.write(str(row['zonei'])+","+str(row['areakey'])+",\""+row['name']+"\","+str(row['east'])+","+str(row['north'])+","+str(row['lon'])+","+str(row['lat'])+"\n")
    #     #end with

    #     print("DeriveAreakeyToZoneCodeFromShapefile discovered " + str(len(dt)) + " zones\n")
    #     return dt
    
################################################################################