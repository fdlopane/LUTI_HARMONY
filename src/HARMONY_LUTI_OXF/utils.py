"""
utils.py
Data building utilities
"""

import numpy as np
import pickle
import struct
import csv

###############################################################################
"""
Load the zone codes lookup from a csv file into a dictionary of dictionaries
"""
#MOVED TO zonecodes.py
# def loadZoneLookup(filename):
#     ZoneLookup = {}
#     with open(filename,'r') as csv_file:
#         reader = csv.reader(csv_file, delimiter=',')
#         header = next(reader,None) #skip header line
#         for row in reader:
#             #zonei,areakey,name,east,north
#             #0,E02000001,City of London 001,532482.7,181269.3
#             zonei = int(row[0])
#             msoa = row[1]
#             name = row[2]
#             east = float(row[3])
#             north = float(row[4])
#             ZoneLookup[msoa] = {
#                 'zonei': zonei,
#                 'name': name,
#                 'east': east,
#                 'north': north
#             }
#             #print("loadZoneLookup: ",msoa,zonei,name,east,north) #DEBUG
#     return ZoneLookup
###############################################################################

"""
Build a trips matrix from data in the Census CSV file (i.e. wu03ew_v2.csv).
The point behind this is to be able to build separate matrices from the sums of a user defined set of columns.
This does assume that the two area keys for the origin destination areas are the first two columns in the file.
NOTE: this defines which way round the i and j are. See comment in code main loop below.
@param name="CSVFilename" string
@param name="ZoneLookup" Lookup between MSOA area key and ZoneCode index number Dictionary of dictionaries
@param name="ColumnNames" Names of columns added together to get the total trips i.e. "UNDERGROUND", "TRAIN", "BUS". These must be in the CSV header line List of string
@returns An observed trips matrix (numpy) matrix
NOTE: you will get a LOT of warnings from this - as long as they're for ODxxx or S92xxx zones, then this is normal
"""
def generateTripsMatrix(CSVFilename, ZoneLookup, ColumnNames):
    N = len(ZoneLookup)
    TijObs = np.zeros(N*N).reshape(N, N)
    for i in range(0,N): # this is basically a hack to absolutely ensure everything is zeroed - above arange sets it to col and row values?
        for j in range(0,N):
            TijObs[i,j]=0.0
    print(ColumnNames,len(ColumnNames))

    with open(CSVFilename) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        fields = next(reader,None)
        # work out column index numbers for all the column names
        ColI = [-1 for i in range(0,len(ColumnNames))] #init with -1
        for i in range(len(fields)):
            field = fields[i]
            for j in range(0,len(ColumnNames)):
                if field == ColumnNames[j]:
                    ColI[j] = i
        print(ColumnNames,ColI)

        # "Area of residence","Area of workplace","All categories: Method of travel to work","Work mainly at or from home","Underground, metro, light rail, tram","Train","Bus, minibus or coach","Taxi","Motorcycle, scooter or moped","Driving a car or van","Passenger in a car or van","Bicycle","On foot","Other method of travel to work"
        # E02000001,E02000001,1506,0,73,41,32,9,1,8,1,33,1304,4
        lineCount = 1
        for row in reader:
            lineCount+=1
            ZoneR = row[0]
            ZoneW = row[1]
            sum = 0
            for i in range(len(ColI)):
                count = int(row[ColI[i]])
                sum += count
            RowR = ZoneLookup.get(ZoneR,'Empty') # could potentially fail if ZoneR or ZoneW didn't exist in the shapefile
            RowW = ZoneLookup.get(ZoneW,'Empty')
            if RowR == 'Empty' or RowW == 'Empty':
                print("Warning: trip " + ZoneR + " to " + ZoneW + " zones not found - skipped")
                continue
            ZoneR_idx = RowR["zonei"]
            ZoneW_idx = RowW["zonei"]
            # TijObs[ZoneR_idx, ZoneW_idx] = sum #this was the original that was apparently the wrong way around
            TijObs[ZoneW_idx, ZoneR_idx] = sum #this is the above line with i and j flipped - right way around
        print('Loaded ', CSVFilename, ' and processed ', lineCount, ' lines of data')
        print('Finished GenerateTripsMatrix')

    return TijObs

###############################################################################

"""
Load a numpy matrix from a file
"""
def loadMatrix(filename):
    with open(filename,'rb') as f:
        matrix = pickle.load(f)
    return matrix

###############################################################################

"""
Save a numpy matrix to a file
"""
def saveMatrix(matrix,filename):
    with open(filename,'wb') as f:
        pickle.dump(matrix,f)

###############################################################################

"""
Load a QUANT format matrix into python.
A QUANT matrix stores the row count (m), column count (n) and then m x n IEEE754 floats (4 byte) of data
"""
def loadQUANTMatrix(filename):
    with open(filename,'rb') as f:
        (m,) = struct.unpack('i', f.read(4))
        (n,) = struct.unpack('i', f.read(4))
        print("loadQUANTMatrix::m=",m,"n=",n)
        matrix = np.arange(m*n,dtype=np.float).reshape(m, n) #and hopefully m===n, but I'm not relying on it
        for i in range(0,m):
            data = struct.unpack('{0}f'.format(n), f.read(4*n)) #read a row back from the stream - data=list of floats
            for j in range(0,n):
                matrix[i,j] = data[j]
    return matrix

###############################################################################

"""
Load CSV data from QUANT in the form of i,j,Oi,Dj,Cij,Tij
The N parameter specifies the matrix size i.e. 7201. Technically we could read
this from the csv if it was complete, but it's easier to pass in
NOTE: the csv data is already in the right format to make training data from, but
this puts it back into a Tij and Cij matrix that we use to build it from again
"""
def loadQUANTCSV(filename,N):
    Tij = np.zeros(N*N).reshape(N,N)
    Cij = np.zeros(N*N).reshape(N,N)
    with open(filename,'r') as f:
        lines=f.readlines() # read everything in to memory at once - it's a lot faster, but there are 52M
        for n in range(1,len(lines)): # need to skip the header
            # so it's i,j,Oi,Dj,Cij,Tij
            fields = lines[n].split(',')
            if len(fields)==6:
                i = int(fields[0])
                j = int(fields[1])
                # Oi = float(fields[2])
                # Dj = float(fields[3])
                valCij = float(fields[4])
                valTij = float(fields[5]) # or int?
                Tij[i,j]=valTij
                Cij[i,j]=valCij
    return Cij, Tij

###############################################################################

"""
Resize a matrix: if smaller then rows and cols are cut out, if bigger then rows and cols are repeated modulo
This is used for the benchmarks, so I can vary the matrix size and see how long it takes
"""
def resizeMatrix(matrix,N):
    (m, n) = np.shape(matrix) # original matrix size
    m2 = np.zeros(N*N).reshape(N, N)
    for i in range(0,N):
        for j in range(0,N):
            m2[i,j]=matrix[i % n,j % n]
    return m2



