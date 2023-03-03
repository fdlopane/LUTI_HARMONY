"""
analytics.py
Produce analytic data for debugging and visualisation
"""

import pandas as pd
from geojson import dump, FeatureCollection, Feature, GeometryCollection, LineString, MultiLineString
from math import sqrt

from HARMONY_LUTI_ATH.globals import *
from HARMONY_LUTI_ATH.utils import loadMatrix


"""
runAnalytics
Function to run the full analytics package
"""
def runAnalytics():
    #produce a flow geojson for the top flows from a probability matrix e.g. retail flows, school flows or hospital flows
    #runAnalyticsRetail(0.004) #was 0.002 then 0.008
    #runAnalyticsSchools(0.008)
    #runAnalyticsHospitals(0.008)
    runAnalyticsJobs(0.008)
    

################################################################################
"""
graphProbabilities - Athens
Produce graph data for the journey to work model
@param threshold The threshold below which to ignore low probability trips
@param dfPointsPopulation MSOA list --> equivalent Attica zones
@param dfPointsZones Point list
@param pointsProbSij matrix of probabilities
@param pointsZonesIDField field name of the unique identifier field in the points files e.g. school id, retail id etc --> ATH zone to zone, not zone to point
@returns a feature collection as a geojson object to be written to file
"""
# def graphProbabilities(threshold,dfPointsPopulation,dfPointsZones,pointsProbSij,pointsZonesIDField): # original code
def graphProbabilities(threshold, dfOriginsPopulation, ProbSij):

    #east,north in retail points zones file (look in zonecodes for the lat lon)
    #east, north and lat,lon in retail points population file

    count=0
    features = []
    m,n = ProbSij.shape
    for i in range(m): #this is the zonei
        row_i = dfOriginsPopulation.loc[dfOriginsPopulation['zonei'] == i]
        i_zone = str(row_i['zone'].values[0])
        i_east = float(row_i['Greek_Grid_east'].values[0])
        i_north = float(row_i['Greek_Grid_north'].values[0])
        #print("graphProbabilities ",i_zone,count)
        # print("graphProbabilities ", i_zone ,"iteration ", i, "of ", m)
        for j in range(n):
            p = ProbSij[i,j]
            if p>=threshold:
                row2 = dfOriginsPopulation.loc[dfOriginsPopulation['zonei'] == j] #yes, zonei==j is correct, they're always called 'zonei'
                j_id = str(row2['zone'].values[0]) #won't serialise a float64 otherwise!
                j_east = float(row2['Greek_Grid_east'].values[0])
                j_north = float(row2['Greek_Grid_north'].values[0])
                the_geom = LineString([(i_east,i_north),(j_east,j_north)])
                f = Feature(geometry=the_geom, properties={"o": i_zone, "d": j_id, "prob":p})
                features.append(f)
                count+=1
            #end if
        #end for
    #end for
    return FeatureCollection(features)


"""
runAnalyticsJobs
@param threshold any probability of trip between MSOA and point below this threshold is cut
"""
def runAnalyticsJobs(threshold):
    dfJobsPointsPopulation = pd.read_csv(data_jobs_population)
    dfJobsPointsZones = pd.read_csv(data_jobs_population) # Yes use data_jobs_population also here as it contains all the zones of interest
    jobs_probSij = loadMatrix(data_jobs_probSij)

    fc = graphProbabilities_jobs(threshold, dfJobsPointsPopulation, dfJobsPointsZones, jobs_probSij, 'msoa') # 'msoa' is the unique identifier of the job MSOA

    #and save fc as a geojson
    with open(os.path.join(modelRunsDir,'analytic_jobs.geojson'), 'w') as f:
        dump(fc, f)

################################################################################

"""
flowArrowsGeoJSON
Take each Aj residential zone and add up the vectors of all the
flows leaving that zone for work in an i zone. This gives you
a residential zone to work zone vector field.
@param Tij the Tij trips matrix to make the flows from
@param The zone codes file as a dataframe, Zone codes from
    zonesdatacoordinates.csv as zonecodes_ATH in main program doesn't have
    the necessary centroid coordinates
@returns a feature collection that you can make a geojson from. This is
    in the Greek grid system
"""
def flowArrowsGeoJSON(Tij,dfZoneCodes):
    #go through all origin zones and find average flow direction
    #print(dfZoneCodes.head())
    #dfZoneCodes.set_index('zonei')
    #make a faster zone lookup as pandas is much too slow
    zonelookup = {}
    for index, row in dfZoneCodes.iterrows():
        zonei = row['zonei']
        east = row['Greek_Grid_east']
        north = row['Greek_Grid_north']
        zonelookup[zonei] = (east,north)
    #end for

    arrowpts = [ [0,0], [0,0.9], [-0.1,0.9], [0,1.0], [0.1,0.9], [0,0.9] ]

    features = []
    m, n = Tij.shape
    for j in range(n): #for all residential zones
        centroidj = zonelookup[j]
        xcj = centroidj[0]
        ycj = centroidj[1]
        dxji=0
        dyji=0
        for i in range(m): #sum all work zone flows to get average flow
            if j==i:
                continue #don't do the self flow
            value=Tij[i,j] #this is flow from originj (residence) to desti (work)
            centroidi = zonelookup[i]
            xci = centroidi[0]
            yci = centroidi[1]
            dx = xci-xcj # j->i vector between centroids - need to normalise this
            dy = yci-ycj
            mag = sqrt(dx*dx+dy*dy)
            #sum normalised direction times value of number of people travelling on link
            dxji+= value * dx/mag
            dyji+= value * dy/mag
        #end for i
        #and make an arrow (xcj,ycj)+(dxji,dyji)*value
        #print("i=",i,"dxji=",dxji,"dyji=",dyji)
        r = sqrt(dxji*dxji+dyji*dyji) #need magnitude of vector as we have to rotate and scale it
        if (r<1): #guard for zero flow, we want it to come out as a dot
            r=1
        #and normalise
        dxji/=r
        dyji/=r
        #now normal to vector j->i
        nxji = dyji
        nyji = -dxji
        ls_pts = [] #to make a linestring
        s = r*1 #scale factor on arrows
        for p in arrowpts:
            #rotated axes are: y along j->i and x along normal(j->i)
            #V = S*AY*(j->i) + S*AX*(Normal(j->i)) where S=scaling, AX,AY=arrow point
            ax = s*p[1]*dxji + s*p[0]*nxji #along ji plus normal
            ay = s*p[1]*dyji + s*p[0]*nyji
            ls_pts.append((xcj+ax,ycj+ay)) #NOTE: east, north fits geojson x,y
        #print(ls_pts)
        the_geom = LineString(ls_pts)
        f = Feature(geometry=the_geom, properties={"originzonei": j})
        features.append(f)
    #end for j

    return FeatureCollection(features)




