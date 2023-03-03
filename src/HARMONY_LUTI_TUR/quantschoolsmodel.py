"""
quantschoolsmodel.py
Build a model for schools.
We have schools and locations (e.g. lat/long) with a pupil capacity.
We have residential zones with numbers of pupils from each educational level.
The aim is to make the flow matrix for how many pupils travel from each zone to each school.
This is done by guessing the alpha and beta factors... (TBC)
School capacity is used as a constraint.
P=Pupils (number per zone)
S=Schools (capacity)
Cij=cost matrix at zone level

It's just a Lakshmanan and Hansen form model.
"""

import numpy as np
import pandas as pd

class QUANTSchoolsModel:
    """
    constructor
    @param n number of residential zones
    @param m number of school point zones
    """
    def __init__(self,m,n):
        #constructor
        self.m = m
        self.n = n
        self.Ei = np.zeros(m)
        self.Aj = np.zeros(n)
        self.cij = np.zeros(1) #costs matrix - set to something BIG later

    ################################################################################

    """
    setPopulationEi
    Given a data frame containing one column with the zone number (i) and one column
    with the actual data values, fill the Ei population property with data so that
    the position in the Ei numpy array is the zonei field of the data.
    The code is almost identical to the setAttractorsAj method.
    NOTE: max(i) < self.m
    """
    def setPopulationEi(self,df,zoneiColName,dataColName):
        df2=df.sort_values(by=[zoneiColName])
        self.Ei = df2[dataColName].to_numpy()
        assert len(self.Ei)==self.m, "FATAL: setPopulationEi length Ei="+str(len(self.Ei))+" MUST equal model definition size of m="+str(self.m)

    ################################################################################

    """
    setAttractorsAj
    Given a data frame containing one column with the zone number (j) and one column
    with the actual data values, fill the Aj attractors property with data so that
    the position in the Aj numpy array is the zonej field of the data.
    The code is almost identical to the setPopulationEi method.
    NOTE: max(j) < self.n
    """
    def setAttractorsAj(self,df,zonejColName,dataColName):
        df2=df.sort_values(by=[zonejColName])
        self.Aj = df2[dataColName].to_numpy()
        assert len(self.Aj)==self.n, "FATAL: setAttractorsAj length Aj="+str(len(self.Aj))+" MUST equal model definition size of n="+str(self.n)

    ################################################################################

    """
    setCostsMatrix
    Assign the cost matrix for the model to use when it runs.
    NOTE: this MUST match the m x n order of the model and be a numpy array
    """

    def setCostMatrixCij(self, cij_0, cij_1, cij_2):
        i, j = cij_0.shape  # Hopefully cij_0.shape = cij_1.shape = cij_2.shape
        assert i == self.m, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.m=" + str(i) + " MUST match model definition of m=" + str(self.m)
        assert j == self.n, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.n=" + str(j) + " MUST match model definition of n=" + str(self.n)
        self.cij_0 = cij_0
        self.cij_1 = cij_1
        self.cij_2 = cij_2

    ################################################################################

    """
    loadSchoolsData
    Loads the schools data from CSV containing [capacity,east,north]
    SchoolCapacity (can be N/A)
    Easting
    Northing
    @param filename Schools file to load (could be primary, middle, high and university)
    @returns DataFrame containing [key,zonei,east,north] and [zonei,capacity] 
    """
    @staticmethod
    def loadSchoolsData(filename):
        df = pd.read_csv(filename,usecols=["zone","capacity","latitude","longitude"])
        df = df.dropna(axis=0) #drop the n/a values
        # REMOVED - all open df = df[df[openFieldName] == 1] #drop any school which is not open (i.e. retain==1)
        df.reset_index(drop=True,inplace=True) #IMPORTANT, otherwise indexes remain for the 28,000 or so rows i.e. idx=0..28000! NOT true row count!

        dfzones = pd.DataFrame({'zone':df['zone'],'zonei':df.index,'latitude':df['latitude'],'longitude':df['longitude']})
        dfattractors = pd.DataFrame({'zonei':df.index,'SchoolCapacity':df['capacity']})

        return dfzones, dfattractors

    ################################################################################

    """
    computeCBar
    Compute average trip length TODO: VERY COMPUTATIONALLY INTENSIVE - FIX IT
    @param Pij trips matrix containing the flow numbers between MSOA (i) and schools (j)
    @param cij trip times between i and j
    """
    @staticmethod
    def computeCBar(Pij,cij):
        CNumerator = np.sum(Pij * cij)
        CDenominator = np.sum(Pij)
        cbar = CNumerator / CDenominator
        return cbar

    ################################################################################

    """
    run Model run3modes_NoCalibration
    Quant model for three modes of transport without calibration
    @returns Sij predicted flows between i and j
    """
    def run3modes_NoCalibration(self, Beta):
        n_modes = len(Beta)  # Number of modes
        print("Running model for ", n_modes, " modes.")
        CBarPred = np.zeros(n_modes)  # initialise CBarPred
        # Initialise variables:
        Sij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty lists equal to n_modes
        cij_k = [self.cij_0, self.cij_1, self.cij_2]  # list of cost matrices

        for k in range(n_modes):  # mode loop
            Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
            ExpMBetaCij = np.exp(-Beta[k] * cij_k[k])
            for i in range(self.m):
                denom = np.sum(self.Aj * ExpMBetaCij[i,])
                for j in range(self.n):
                    Sij[k][i, j] = self.Ei[i] * (self.Aj[j] * ExpMBetaCij[i, j]) / denom

        for k in range(n_modes):
            CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])

        return Sij, CBarPred

    ################################################################################

    """
    computeProbabilities3modes
    Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones
    """
    def computeProbabilities3modes(self, Sij):
        n_modes = 3
        probSij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty list equal to n_modes

        for k in range(n_modes):
            probSij[k] = np.arange(self.m * self.n, dtype=np.float).reshape(self.m, self.n)
            for i in range(self.m):
                sum = np.sum(Sij[k][i,])
                if sum <= 0:
                    sum = 1  # catch for divide by zero - just let the zero probs come through to the final matrix
                probSij[k][i,] = Sij[k][i,] / sum

        return probSij