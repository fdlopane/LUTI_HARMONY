"""
analytics.py
Produce analytic data for debugging and visualisation
"""
from geojson import dump, FeatureCollection, Feature, GeometryCollection, LineString, MultiLineString
from math import sqrt

    

################################################################################

"""
graphProbabilities - OXF
Produce graph data for the schools and hospitals model
@param threshold The threshold below which to ignore low probability trips
@param dfPointsPopulation zone list
@param dfPointsZones Point list
@param pointsProbSij matrix of probabilities
@param pointsZonesIDField field name of the unique identifier field in the points files e.g. school id, hospital id etc
@returns a feature collection as a geojson object to be written to file (probably)
"""
def graphProbabilities_GB(threshold,dfPointsPopulation,dfPointsZones,pointsProbSij,pointsZonesIDField):

    #east,north in retail points zones file (look in zonecodes for the lat lon)
    #east, north and lat,lon in retail points population file

    count=0
    features = []
    m,n = pointsProbSij.shape
    for i in range(m): #this is the zonei
        row_i = dfPointsPopulation.loc[dfPointsPopulation['zonei'] == i]
        i_msoaiz = row_i['msoaiz'].values[0]
        i_east = float(row_i['osgb36_east'].values[0])
        i_north = float(row_i['osgb36_north'].values[0])
        #print("graphProbabilities ",i_msoaiz,count)
        print("graphProbabilities ", i_msoaiz ,"iteration ", i, "of ", m)
        for j in range(n):
            p = pointsProbSij[i,j]
            if p>=threshold:
                row2 = dfPointsZones.loc[dfPointsZones['zonei'] == j] #yes, zonei==j is correct, they're always called 'zonei'
                j_id = str(row2[pointsZonesIDField].values[0]) #won't serialise a float64 otherwise!
                j_east = float(row2['east'].values[0])
                j_north = float(row2['north'].values[0])
                the_geom = LineString([(i_east,i_north),(j_east,j_north)])
                f = Feature(geometry=the_geom, properties={"o": i_msoaiz, "d": j_id, "prob":p})
                features.append(f)
                count+=1
            #end if
        #end for
    #end for
    return FeatureCollection(features)

"""
graphProbabilities - OXF
Produce graph data for the journey to work model
@param threshold The threshold below which to ignore low probability trips
@param dfPointsPopulation MSOA list --> equivalent Attica zones
@param dfPointsZones Point list
@param pointsProbSij matrix of probabilities
@param pointsZonesIDField field name of the unique identifier field in the points files e.g. school id, retail id etc --> OXF zone to zone, not zone to point
@returns a feature collection as a geojson object to be written to file
"""
# def graphProbabilities(threshold,dfPointsPopulation,dfPointsZones,pointsProbSij,pointsZonesIDField): # original code
def graphProbabilities(threshold, dfOriginsPopulation, ProbSij):


    count=0
    features = []
    m,n = ProbSij.shape
    for i in range(m): #this is the zonei
        row_i = dfOriginsPopulation.loc[dfOriginsPopulation['zonei'] == i]
        i_zone = str(row_i['zone'].values[0])
        i_east = float(row_i['Easting'].values[0])
        i_north = float(row_i['Northing'].values[0])
        #print("graphProbabilities ",i_zone,count)
        # print("graphProbabilities ", i_zone ,"iteration ", i, "of ", m)
        for j in range(n):
            p = ProbSij[i,j]
            if p>=threshold:
                row2 = dfOriginsPopulation.loc[dfOriginsPopulation['zonei'] == j] #yes, zonei==j is correct, they're always called 'zonei'
                j_id = str(row2['zone'].values[0]) #won't serialise a float64 otherwise!
                j_east = float(row2['Easting'].values[0])
                j_north = float(row2['Northing'].values[0])
                the_geom = LineString([(i_east,i_north),(j_east,j_north)])
                f = Feature(geometry=the_geom, properties={"o": i_zone, "d": j_id, "prob":p})
                features.append(f)
                count+=1
            #end if
        #end for
    #end for
    return FeatureCollection(features)


################################################################################

"""
flowArrowsGeoJSON
Take each Aj residential zone and add up the vectors of all the
flows leaving that zone for work in an i zone. This gives you
a residential zone to work zone vector field.
@param Tij the Tij trips matrix to make the flows from
@param The zone codes file as a dataframe, Zone codes from
    zones_data_coordinates.csv as zonecodes_TUR in main program doesn't have
    the necessary centroid coordinates
@returns a feature collection that you can make a geojson from. This is
    in the Italian grid system
"""
def flowArrowsGeoJSON(Tij,dfZoneCodes):
    #go through all origin zones and find average flow direction
    #print(dfZoneCodes.head())
    #dfZoneCodes.set_index('zonei')
    #make a faster zone lookup as pandas is much too slow
    zonelookup = {}
    for index, row in dfZoneCodes.iterrows():
        zonei = row['zonei']
        east = row['Easting']
        north = row['Northing']
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




