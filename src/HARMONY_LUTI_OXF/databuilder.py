"""
databuilder.py

Build data for project
At the moment, it builds model-runs/retailpoints_geocoded.csv with oa geocoded Geolytix retail points
"""
import os
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
import csv

from HARMONY_LUTI_OXF.globals import *
from HARMONY_LUTI_OXF.utils import loadQUANTMatrix
from HARMONY_LUTI_OXF.zonecodes import ZoneCodes

"""
ensureFile
Check the local file system for the existence of a file and download it from
the givel url if it's not present. Used for installation of data file from
remote sources so we don't check "official" data into the GitHub repo.
@param localFilename The file to check for existence on the local file system
@param url Where to download it if it's not there
"""
def ensureFile(localFilename,url):
    if not os.path.isfile(localFilename):
        print("Downloading "+localFilename+" from "+url)
        #wget.download(url,localFilename) # won't allow headers to add user agent

        #use requests module instead, which doesn't set the file info like wget does - datestamp wrong!

        headers={'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, allow_redirects=True, headers=headers)
        open(localFilename, 'wb').write(r.content)

################################################################################

"""
geocodeGeolytix
Geocode the geolytix point file by appending an OA code and an MSOA code.
Write out csv file containing the geocoded geolytix data.
The file name consts below are defined in globals.py
PRE:
  requires "data_oa_shapefile" which points to the infuse OA boundary file for E, W, S and NI
  requires "data_msoa_shapefile" similarly for the MSOA boundary file
  requires "data_geolytix_retailpoints" containing the Geolytix Retail Points csv file
POST:
  generates "data_retailpoints_geocoded" containing the Geolytix retail point data geocoded with OA and MSOA
"""
def geocodeGeolytix():
    #read OA shapefile
    print("loading oa shapefile from: ",data_oa_shapefile)
    oa_boundary = gpd.read_file(data_oa_shapefile)
    print(data_oa_shapefile,len(oa_boundary),"rows")
    #print(oa_boundary.head(6))
    #print(oa_boundary.shape)
    # plot the data using geopandas .plot() method - TAKES AGES THOUGH!
    #fig, ax = plt.subplots(figsize = (10,10))
    #oa_boundary.plot(ax=ax)
    #plt.show()

    #read MSOA shapefile
    print("loading msoa shapefile from: ",data_msoa_shapefile)
    msoa_boundary = gpd.read_file(data_msoa_shapefile)
    print(data_msoa_shapefile,len(msoa_boundary),"rows")

    #read geolytix retail data
    #id,retailer,fascia,store_name,add_one,add_two,town,suburb,postcode,long_wgs,lat_wgs,bng_e,bng_n,pqi,open_date,size_band
    #1010004593,Tesco,Tesco Express,Tesco Eastern Seaside Road Express,133-135 Seaside Road,,Eastbourne,Meads,BN21 3PA,0.293276318,50.76901815,561808.4009,99110.41946,Rooftop geocoded by Geolytix,,"< 3,013 ft2 (280m2)"
    #size bands: "< 3,013 ft2 (280m2)" "15,069 < 30,138 ft2 (1,400 < 2,800 m2)" "3,013 < 15,069 ft2 (280 < 1,400 m2)" "30,138 ft2 > (2,800 m2)"
    medianSizeBands = {
        "< 3,013 ft2 (280m2)": 140.0,
        "3,013 < 15,069 ft2 (280 < 1,400 m2)": 840.0,
        "15,069 < 30,138 ft2 (1,400 < 2,800 m2)": 2100.0,
        "30,138 ft2 > (2,800 m2)": 2800.0
    }
    retailPoints = []
    print("loading retail points from: ",data_geolytix_retailpoints)
    with open(data_geolytix_retailpoints, newline = '') as csvFile:
        reader = csv.reader(csvFile,delimiter=',',quotechar='"')
        next(reader) #skip header row
        for row in reader:
            fascia = row[2] # e.g. Tesco Express
            east = float(row[11])
            north = float(row[12])
            strFloorspace = row[15] # text version
            floorspace = medianSizeBands[strFloorspace]
            point = Point(east,north)
            retailPoints.append({'name':fascia,'point':point,'floorspace':floorspace})
    print(data_geolytix_retailpoints,len(retailPoints),"rows")

    # point in polygon on the oa boundaries using geopandas in-built spatial index to find out which oa and msoa each retail point is in
    oa_retail_count = {} # keyed on oa, contains count of how many retail points in each oa
    retail_oa_zone_code = {} # keyed on retailPoints index id, stores the od code this point is in i.e. it's a join to the retailPoints data frame
    sindex_oa = oa_boundary.sindex

    msoa_retail_count = {}
    retail_msoa_zone_code = {}
    sindex_msoa = msoa_boundary.sindex

    for retail_idx, retailer in enumerate(retailPoints):
        rp = retailer['point']
        # oa point in polygon
        possible_matches_index_oa = list(sindex_oa.intersection(rp.bounds)) # coarse match using the OA spatial index
        possible_matches_oa = oa_boundary.iloc[possible_matches_index_oa] # get actual geom back from indexes just returned
        precise_matches_oa = possible_matches_oa[possible_matches_oa.intersects(rp)] # perform expensive precise match using geom
        # msoa point in polygon
        possible_matches_index_msoa = list(sindex_msoa.intersection(rp.bounds)) #c oarse match using the MSOA spatial index
        possible_matches_msoa = msoa_boundary.iloc[possible_matches_index_msoa] # get actual geom back from indexes just returned
        precise_matches_msoa = possible_matches_msoa[possible_matches_msoa.intersects(rp)] # perform expensive precise match using geom

        count_oa = len(precise_matches_oa)
        count_msoa = len(precise_matches_msoa)
        # assert... should never happen
        if count_oa>1 or count_msoa>1:
            print("error: retail point in more than one output area or msoa: ",retailer,precise_matches_oa, precise_matches_msoa)
        # end assert
        # assert... of course you could also check that precise_matches isn't empty as we have points for channel isles and isle of man, but no OAs
        if count_oa==0 or count_msoa==0:
            print("count=0 for ",rp)
            continue
        # end second assert
        match_oa = precise_matches_oa.iloc[0] # ugly
        match_msoa = precise_matches_msoa.iloc[0] # ugly
        oa_code = match_oa.geo_code
        msoa_code = match_msoa.geo_code
        if oa_code in oa_retail_count:
            oa_retail_count[oa_code]+=1 # increment exising key
        else:
            oa_retail_count[oa_code]=1 # new key
        retail_oa_zone_code[retail_idx]=oa_code # store the oa that this retail point sits inside by index
        # and now for the msoa
        if msoa_code in msoa_retail_count:
            msoa_retail_count[msoa_code]+=1 # increment exising key
        else:
            msoa_retail_count[msoa_code]=1 # new key
        retail_msoa_zone_code[retail_idx]=msoa_code # store the oa that this retail point sits inside by index

    print('oa_retail_count contains: ',len(oa_retail_count),'separate output areas containing retail points')
    print('msoa_retail_count contains: ',len(msoa_retail_count),'separate mid layer super output areas containing retail points')
    print('total retail points: ',len(retailPoints))

    # Write out a retail points file with oa attached to the retail points
    with open(data_retailpoints_geocoded, 'w', newline = '') as csvFile:
        writer = csv.writer(csvFile,delimiter=',')
        writer.writerow(['name','east','north','floorspace','oa','msoa'])
        for idx, rp in enumerate(retailPoints):
            if idx in retail_oa_zone_code:
                writer.writerow([ rp['name'], rp['point'].x, rp['point'].y, rp['floorspace'], retail_oa_zone_code[idx], retail_msoa_zone_code[idx] ])

################################################################################

"""
Compute cost function for Geolytix retail points based on the QUANT MSOA to MSOA costs matrix
with a correction based on how far the Geolytix point is from the MSOA centroid.
Costs are times in minutes.
The Geolytix points are keyed using the store name, east and north locations.
The cost matrix used is for the road mode.
PRE: requires geocodeGeolytix to have been called first to generate the Geolytix MSOA file
    Consumes QUANT cost matrix file, centroids file and geolytix geocoded retail points file
POST: creates a GeolytixCost file in the modelruns dir: data_retailpoints_msoa_costs
"""
def computeGeolytixCosts():
    # const to define what speed we travel the additional distance to the retail point e.g. 30mph = 13 ms-1
    metresPerSec = 13.0

    # load cost matrix, time in minutes between MSOA zones
    cij = loadQUANTMatrix(os.path.join(modelRunsDir,QUANTCijRoadMinFilename))
    m, n = cij.shape

    # load zone codes lookup file to convert MSOA codes into zone i indexes for the model
    zonecodes = ZoneCodes.fromFile()

    # load the centroids file - this gives us the points used for the msoa trips in cij
    # zonecode,zonei,zone_lat,zone_lon,vertexid,vertex_lat,vertex_lon,distMetres
    # E02000001,0,51.515,-0.09051585,532370_181559,51.5174481287716,-0.09364236153270947,348.1855
    df = pd.read_csv(os.path.join(modelRunsDir,QUANTCijRoadCentroidsFilename))
    # df['geometry'] = df.apply(lambda row: Point(row.vertex_lon, row.vertex_lat), axis=1)
    origin_crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(df, crs=origin_crs, geometry=gpd.points_from_xy(df.vertex_lon, df.vertex_lat))
    # This is in lat/lon when we want east north
    centroids = gdf.to_crs("EPSG:27700")
    dest_unary = centroids["geometry"].unary_union # and need this join for the centroid points nearest lookup

    with open(data_retailpoints_msoa_costs, 'w') as writer:
        writer.write("origin,destination,cost_minutes,retail_delta_mins\n")
        # open the geocoded Geolytix file containing the retail points with their MSOA codes
        # name,east,north,floorspace,oa,msoa
        # Tesco Express,561808.4009,99110.41946,140.0,E00106155,E02004365
        count=0
        with open(data_retailpoints_geocoded, 'r', newline = '') as csvFile:
            reader = csv.reader(csvFile,delimiter=',')
            next(reader) # skip header row
            for row in reader:
                name = row[0]
                print(str(count)+" "+name)
                count+=1
                east = float(row[1])
                north = float(row[2])
                # msoacode = row[5]
                retailkey = name+"_"+str(int(east))+"_"+str(int(north))
                # lookup the msoa data
            
                # lookup using msoa code - but Scotland IZ is 2001 when we have 2011!
                # print("looking up msoa code ",msoacode)
                # centroidrow = centroids.loc[centroids['zonecode'] == msoacode]
                # zonei = int(centroidrow.zonei)
                # p = centroidrow.geometry
            
                # lookup by closest msoa centroid to geolytix point
                near = nearest_points(Point(east,north),dest_unary)
                match_geom = centroids.loc[centroids.geometry==near[1]]
                zonei = int(match_geom.zonei)
                p = match_geom.geometry
                # print(match_geom)

                east2=float(p.centroid.x)
                north2=float(p.centroid.y)

                dx = east-east2
                dy = north-north2
                dist = np.sqrt(dx*dx+dy*dy) # dist between retail point and centroid used for shortest path
                # work out an additional delta cost based on increased time getting from this point to the centroid
                deltaCost = (dist/metresPerSec)/60.0 # transit time in mins

                # Now write out this Geolytix point to every MSOA in turn in both directions, adding on the deltaCost
                # NOTE: I'm only going to output msoa zone to retail point to halve the file size
                for zonecodej in zonecodes.dt:
                    zonej = zonecodes.dt[zonecodej]['zonei']
                    costij = cij[zonei,zonej]
                    costji = cij[zonej,zonei]
                    # retail to msoa centroid
                    # writer.write("{0}, {1}, {2:.2f}, {3:.2f}\n".format(retailkey,zonecodej,(costij+deltaCost),deltaCost))
                    # msoa centroid to retail
                    writer.write("{0}, {1}, {2:.2f}, {3:.2f}\n".format(zonecodej,retailkey,(costji+deltaCost),deltaCost))
                #end for
            #end for
        #end with (reader)
    #end with (writer)

################################################################################

"""
geolytixRegression
Yet another function mangling the Geolytix data into something else. This time it's
to take the restricted file containing the floorspace and annual takings data that
we got directly from them and aren't allowed to release and merge it into the
open data. This is done by doing a linear regression of "modelled sq ft" to
"Modelled turnover annual" on the restricted data and then applying that to
the open data with the four levels of floorspace in order to estimate
annual turnover. It's probably a really bad guess, but then it's retail data.
Regression model is a scikit linear regression class.
@param inRestrictedFilename
@param inOpenDataFilename
@param outGeolytixRegression
"""
def geolytixRegression(inRestrictedFilename,inOpenDataFilename,outGeolytixRegression):
    dfRestricted = pd.read_csv(inRestrictedFilename)
    # doing this very naively, take the "modelled sq ft" column and use it to
    # predict "Modelled turnover annual"
    x = np.array(dfRestricted['modelled sq ft'].tolist()).reshape(-1,1) # it's an nx1 array of 1d data
    y = np.array(dfRestricted['Modelled turnover annual'].tolist())
    model = LinearRegression().fit(x,y)
    print('dataBuilder::geolytixRegression:: model.score = ',model.score(x,y))
    # y_new = model.predict(new_x)
    #TODO: can you print out the model params here?
    print("REGRESSION DATA")
    print(model.get_params())
    print(model.coef_) # [738.96967827] #NOTE: this is an np.array
    print(model.intercept_) # 2429081.7748885565
    print(dir(model))
    print(dir(model.coef_))
    print(model.coef_.T)
    print("END OF REGRESSION DATA")

    # using our new found ability to predict annual turnover from floorspace,
    # go through all the open data, drop in the real data from the restricted
    # file where the unique ids match, or use the regression model to fill in
    # the turnover from the approximate floorspace range otherwise.
    dfOpen = pd.read_csv(inOpenDataFilename)

    # id,retailer,fascia,store_name,add_one,add_two,town,suburb,postcode,long_wgs,lat_wgs,bng_e,bng_n,pqi,open_date,size_band
    # 1010004593,Tesco,Tesco Express,Tesco Eastern Seaside Road Express,133-135 Seaside Road,,Eastbourne,Meads,BN21 3PA,0.293276318,50.76901815,561808.4009,99110.41946,Rooftop geocoded by Geolytix,,"< 3,013 ft2 (280m2)"
    # size bands: "< 3,013 ft2 (280m2)" "15,069 < 30,138 ft2 (1,400 < 2,800 m2)" "3,013 < 15,069 ft2 (280 < 1,400 m2)" "30,138 ft2 > (2,800 m2)"
    medianSizeBands = {
        "< 3,013 ft2 (280m2)": 1506.5,
        "3,013 < 15,069 ft2 (280 < 1,400 m2)": 9041.0,
        "15,069 < 30,138 ft2 (1,400 < 2,800 m2)": 22603.5,
        "30,138 ft2 > (2,800 m2)": 30138.0
    }
    dfOpen['modelled sq ft'] = [ medianSizeBands[f] for f in dfOpen['size_band'] ] # that fills a new sq ft column with approx sizes
    # now take real sq ft sizes from the restricted data where ids match - CAN SKIP this stage if restricted data not available
    # NOTE: key on restricted data is 'gluid', while on open data it's 'id'
    # this is the easy way to do it - make the accurate data into a dict [gluid,value], then iterate replace cells in the open data
    dict_accurate_floorspace = dfRestricted.set_index('gluid').to_dict()['modelled sq ft'] #key is gluid
    for f in dict_accurate_floorspace:
        dfOpen.loc[dfOpen['id']==f, 'modelled sq ft'] = dict_accurate_floorspace[f] #key is id
    # That's sorted the floorspace, now put in a modelled turnover column
    # Use the regression model to fill in first
    dfOpen['Modelled turnover annual'] = [model.predict(np.array(x).reshape(-1,1))[0] for x in dfOpen['modelled sq ft'] ]
    # Take the real turnover figures from the restricted data where ids match - CAN SKIP this stage if restricted data not available
    dict_accurate_turnover = dfRestricted.set_index('gluid').to_dict()['Modelled turnover annual'] #key is gluid
    for t in dict_accurate_turnover:
        dfOpen.loc[dfOpen['id']==t, 'Modelled turnover annual'] = dict_accurate_turnover[t] #key is id
    # dfOpen['Modelled turnover annual'] = [ (dict_accurate_turnover if i in dict_accurate_turnover else dfOpen['Modelled turnover annual'] ) for i in dfOpen['id'] ]
    # And that's all there is, except to point out that there could be gluids in the restricted file that aren't in the open data - ignoring the
    # extras for now - I did find the odd one, but then there are 16,000 retail points here

    # and save:
    dfOpen.to_csv(outGeolytixRegression)

################################################################################

"""
geolytixOpenDataRegression
Same as geolytixRegression, but only uses the Geolytix open data so that it's repeatable.
The regression params are hardcoded, but based on values obtained from running our own
regression on restricted Geolytix data.
What this does is to take the floorspace bands in the Geolytix open data, which are
aggregated into one of four bands, and write a new "modelled sq ft" column and a new
"Modelled turnover annual" column based on regression data. It's just a simple linear
regression from the floorspace to turnover, based on hard coding regression data which
we obtained from the restricted dataset containing the economic data for 4,000 retail
points.
@param inOpenDataFilename
@param outGeolytixRegression 
"""
def geolytixOpenDataRegression(inOpenDataFilename,outGeolytixRegression):
    # set up a linear regression predictor directly from data we have obtained from restricted sales data
    model = LinearRegression()
    model.coef_=np.array([738.96967827])
    model.intercept_=2429081.7748885565
    model.copy_X=True
    model.fit_intercept=True
    model.normalize=False
    # and this is default: {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}
    
    # the rest is just a copy from geolytixRegression... but with the merge of the restricted data and open data removed - everything comes from the regression params

    # using our new found ability to predict annual turnover from floorspace,
    # go through all the open data, drop in the real data from the restricted
    # file where the unique ids match, or use the regression model to fill in
    # the turnover from the approximate floorspace range otherwise.
    dfOpen = pd.read_csv(inOpenDataFilename)

    # id,retailer,fascia,store_name,add_one,add_two,town,suburb,postcode,long_wgs,lat_wgs,bng_e,bng_n,pqi,open_date,size_band
    # 1010004593,Tesco,Tesco Express,Tesco Eastern Seaside Road Express,133-135 Seaside Road,,Eastbourne,Meads,BN21 3PA,0.293276318,50.76901815,561808.4009,99110.41946,Rooftop geocoded by Geolytix,,"< 3,013 ft2 (280m2)"
    # size bands: "< 3,013 ft2 (280m2)" "15,069 < 30,138 ft2 (1,400 < 2,800 m2)" "3,013 < 15,069 ft2 (280 < 1,400 m2)" "30,138 ft2 > (2,800 m2)"
    medianSizeBands = {
        "< 3,013 ft2 (280m2)": 1506.5,
        "3,013 < 15,069 ft2 (280 < 1,400 m2)": 9041.0,
        "15,069 < 30,138 ft2 (1,400 < 2,800 m2)": 22603.5,
        "30,138 ft2 > (2,800 m2)": 30138.0
    }
    dfOpen['modelled sq ft'] = [ medianSizeBands[f] for f in dfOpen['size_band'] ] #that fills a new sq ft column with approx sizes
    # That's sorted the floorspace, now put in a modelled turnover column
    # use the regression model to fill in turnover
    dfOpen['Modelled turnover annual'] = [model.predict(np.array(x).reshape(-1,1))[0] for x in dfOpen['modelled sq ft'] ]
    # And that's all there is, except to point out that there could be gluids in the restricted file that aren't in the open data - ignoring the
    # extras for now - I did find the odd one, but then there are 16,000 retail points here

    # and save:
    dfOpen.to_csv(outGeolytixRegression)

################################################################################

"""
buildSchoolsPopulationTable
Takes the QS103 age structure by single age Census table and adds up the number of
primary and secondary age school children by MSOA.
Primary age: 5-11
Secondary age: 12-16
POST: writes out data_schoolagepopulation_englandwales
NOTE: you then need to join the england/wales and scotland files together
"""
def buildSchoolsPopulationTableEnglandWales(input_census_data):
    df = pd.read_csv(input_census_data)
    df['count_primary'] = (df['Age: Age 5; measures: Value']
        + df['Age: Age 6; measures: Value'] + df['Age: Age 7; measures: Value'] + df['Age: Age 8; measures: Value']
        + df['Age: Age 9; measures: Value'] + df['Age: Age 10; measures: Value'] + df['Age: Age 11; measures: Value']
    )
    df['count_secondary'] = (df['Age: Age 12; measures: Value'] + df['Age: Age 13; measures: Value'] + df['Age: Age 14; measures: Value']
        + df['Age: Age 15; measures: Value'] + df['Age: Age 16; measures: Value']
    )
    df2 = pd.DataFrame({'geography code': df['geography code'], 'count_primary': df['count_primary'], 'count_secondary': df['count_secondary']})
    df2.to_csv(data_schoolagepopulation_englandwales)

################################################################################

"""
buildSchoolsPopulationTableScotland
Build schools population table for scotland based on QS103SC from the census bulk download:
https://www.scotlandscensus.gov.uk/ods-web/data-warehouse.html#bulkdatatab
This is from the data zones 2001 file, NOT the datazones 2011 file.
Also required is the 2001 DZ to 2001 IZ conversion table.
NOTE: the QS103SC file had to have the "-" chars turned into zeroes, otherwise pandas won't
recognise the age columns as integers and defaults to strings, after which strange things happen
POST: writes out data_schoolagepopulation_scotland
NOTE: you then need to join the england/wales and scotland files together
"""
def buildSchoolsPopulationTableScotland(input_census_data, lookup_census_data):
    df = pd.read_csv(input_census_data) # join on "Unnamed: 0", it's blank! This is the datazone code field
    df.set_index('Unnamed: 0')
    dfLookup = pd.read_csv(lookup_census_data) # join on ZONECODE, which is the datazone code
    df = df.join(other=dfLookup.set_index('ZONECODE'),on='Unnamed: 0')
    # df.to_csv("scotlookup_hack.csv")
    # sum age columns to get the primary and secondary count columns - reducing columns makes it easier later
    df['count_primary'] = df['5'] + df['6'] + df['7'] + df['8'] + df['9'] + df['10'] + df['11']
    df['count_secondary'] = df['12'] + df['13'] + df['14'] + df['15'] + df['16']
    # df.to_csv("scotlookup_hack2.csv")
    # That's the age ranges summed to give the primary and secondary counts on the DZ geomgraphy,
    # now aggregate and sum the DZ codes to make an IZ dataset
    # group and sum on the IZ_CODE field
    df2 = df.groupby(['IZ_CODE']).agg({'count_primary': "sum", 'count_secondary': "sum"})
    df2.to_csv(data_schoolagepopulation_scotland)

################################################################################

"""
buildTotalPopulationTable
Takes the QS103 age structure and merges the England/Wales and Scotland data together.
The added complication is that Scotland is on DZ2001 boundaries, so you have to group.
NOTE: this is basically a merge of the england and scotland version of the school age
population table creation above. Only it's only using a single total people column.
Then it takes the England+Wales and Scotland tables and merges them together to
produce an Age Population table containing all the individual age counts. This is
used for the hospitals age model.
POST: writes out data_totalpopulation and data_agepopulation
"""
def buildTotalPopulationTable(censusEW, censusSC, lookupSC):
    # England and Wales
    dfEW = pd.read_csv(censusEW)
    dfEW['count_allpeople'] = dfEW['Age: All categories: Age; measures: Value'] # you could just rename the col, not copy
    dfEW2 = pd.DataFrame({'geography code': dfEW['geography code'], 'count_allpeople': dfEW['count_allpeople']})
    
    # Scotland (DZ2001)
    dfS = pd.read_csv(censusSC) # join on "Unnamed: 0", it's blank! This is the datazone code field
    dfS.set_index('Unnamed: 0')
    dfSLookup = pd.read_csv(lookupSC) # join on ZONECODE, which is the datazone code
    dfS = dfS.join(other=dfSLookup.set_index('ZONECODE'),on='Unnamed: 0')
    dfS['count_allpeople'] = dfS['All people']
    dfS2 = dfS.groupby(['IZ_CODE']).agg({'count_allpeople': "sum"})
    dfS3 = pd.DataFrame({'msoaiz': dfS2.index, 'count_allpeople': dfS2['count_allpeople'] }) # cols are weird if you don't do this - dfS2 has iz as the index and we need a col
    # NOTE: I deleted the line in the Scotland QS103SC file which contains the total for the whole of Scotland - it's a pain

    # Now merge dfEW2 and dfS3 into one table
    dfEW2.reset_index()
    dfS3.reset_index()
    dfEW2.columns=['msoaiz','count_allpeople']
    dfEWS = dfEW2.append(dfS3)
    dfEWS.to_csv(data_totalpopulation,index=False) # drop the index col off as EW has a numberic row id and scot has idx=IZ code - why on earth??

    # Now repeat, but adding in the age data to make an age structure table
    # Scotland is a problem... we have to aggregate all these columns by IZ groups of DZ entries
    # ,All people,Under 1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100 and over
    dfSA = dfS.groupby(['IZ_CODE']).sum()
    dfSA.index.rename("msoaiz", inplace=True)
    dfSA['msoaiz']=dfSA.index #have to copy index into regular field, otherwise I can't merge it with the EW data
    dfSA.drop({'EASTING', 'NORTHING', 'MMW_CODE', 'SPC07_CODE', 'LA_CODE','SGUR2010_6FOLD', 'LEC_CODE', 'SETT2008_CODE'},axis=1,inplace=True) #and drop the Scotland specific columns
    # dfSA.to_csv(data_agepopulation,index=True)
    # now you need to merge in England and Wales
    dfEW.reset_index()
    dfEW.drop({'Rural Urban'},axis=1,inplace=True) # drop this col as not in Scotland data - what is it anyway, all it contains is 'Total'?
    dfEW.rename({'geography code': 'msoaiz'},axis=1,inplace=True)
    dfEW.rename({'Age: All categories: Age; measures: Value': 'All people'},axis=1,inplace=True)
    dfEW.rename({'Age: Age under 1; measures: Value': 'Under 1'},axis=1,inplace=True)
    for i in range(1,100):
        dfEW.rename({'Age: Age '+str(i)+'; measures: Value': str(i)},axis=1,inplace=True)
    dfEW.rename({'Age: Age 100 and over; measures: Value': '100 and over'},axis=1,inplace=True)
    # That gets all the age column names the same, now we can append them together NOTE: there are some column differences between the two data
    dfAgeEWS = dfEW.append(dfSA)
    dfAgeEWS.to_csv(data_agepopulation,index=False)

################################################################################

"""
changeGeography
Take data which is on geomInput and convert it to geomOutput by working out
how much of each area in geomOutput is covered by areas in geomInput and
doing a proportional area (in->out) on the data to work out a data value for
the target area in geomOutput.
@param geomInput Geometry input filename (shapefile)
@param inAreaKeyAttrName Name of the attribute in the input shapefile containing the area key
@param geomOutput Geometry output filename (shapefile) NOTE: we don't write out anything to this file!
@param outAreaKeyAttrName Name of the attribute in the output shapefile contatining the area key
@param datafile csv file with area codes that match "geomInput" and a column labelled "dataFieldName"
@param dataAreaKeyFieldName
@param dataFieldName the name of the field in "data" containing the data to be converted
@returns a data frame with keycodes from geomOutput, but data from the original "data" file converted
"""
def changeGeography(geomInput,inAreaKeyAttrName,geomOutput,outAreaKeyAttrName,datafile,dataAreaKeyFieldName,dataFieldName):
    gdfIn = gpd.read_file(geomInput)
    gdfOut = gpd.read_file(geomOutput)
    #todo: need a sanity check that they're in the same projection!
    df = pd.read_csv(datafile)
    for outPoly in gdfOut.itertuples(index = False):
        outAreaKey = getattr(outPoly, outAreaKeyAttrName)
        fullArea = outPoly.geometry.area
        value = 0.0
        for inPoly in gdfIn.itertuples(index=False):  #so much for spatial indexing - bubblesort!
            if (outPoly.geometry.intersects(inPoly.geometry)):
                try:
                    coverageGeom = outPoly.geometry.intersection(inPoly.geometry) # catch here?
                    subAreaKey = getattr(inPoly,inAreaKeyAttrName)
                    subData = df.loc[df[dataAreaKeyFieldName]==subAreaKey,dataFieldName].values[0]
                    subArea = coverageGeom.area
                    value=value + subArea/fullArea*subData
                except:
                    print("Geometry error on "+outAreaKey)
        print(outAreaKey+","+str(value))

################################################################################

"""
matchHospitalEpisodeData
We have two tables for hospitals: the NHS site data and episodes and admissions data
containing numbers for different age groups plus male/female. The hospital codes
don't match.
Attempt to match up one table to the other one based on the plain text name so that
we can add the location code to the episode by age table.
NOTE: the main site database has 1100 lines but the age table only has 470.
"""
def matchHospitalEpisodeData():
    dfHospitals = pd.read_csv(data_hospitals) # NOTE: this isn't the full table but a subset
    dfHospitalsAgeEpisodes = pd.read_csv(data_hospitalEpisodes)

    # The codes don't match on these tables (great), so let's try a fuzzy match on names (!)

    # The most obvious way of doing this is to load it all into memory rather than mess about iterating dataframes
    hospitalNameList = dfHospitals['Site Name'].tolist()
    hospitalNameList = [x.lower() for x in hospitalNameList]
    bagOfWords = {}
    for name in hospitalNameList:
        words = name.split()
        for word in words:
            if word in bagOfWords:
                bagOfWords[word]+=1
            else:
                bagOfWords[word]=1

    # todo: do you need a normalise probs here?

    # That gets us word probabilities, now let's try matching
    for row in dfHospitalsAgeEpisodes.itertuples(index=False):
        name1 = row.description.lower()
        words1 = name1.split()
        maxScore=0
        maxName=''
        for k in hospitalNameList:
            score=0
            words2 = k.split()
            for w1 in words1: # not exactly elegant, bubble on w1,w2, but hash is probably overkill
                for w2 in words2: # OK, I know it matches twice
                    if (w1==w2):
                        score=score+1/bagOfWords[w2]
            if score>maxScore:
                maxScore=score
                maxName=words2
        print(maxScore,',',words1,',matches,',maxName)