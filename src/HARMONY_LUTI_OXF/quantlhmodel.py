"""
QUANTLHModel.py

Lakshmanan and Hansen form model. Used as a base for other models.
Uses an attractor, Aj, Population Ei and cost matrix Cij.
"""

import numpy as np

class QUANTLHModel:
    """
    constructor
    @param n number of residential zones (MSOA)
    @param m number of school point zones
    """

    def __init__(self, m, n):
        # constructor
        self.m = m
        self.n = n
        self.Ei = np.zeros(m)
        self.Aj = np.zeros(n)
        self.cij_0 = np.zeros(1)  # costs matrix - set to something BIG later
        self.cij_1 = np.zeros(1)  # costs matrix - set to something BIG later
        self.cij_2 = np.zeros(1)  # costs matrix - set to something BIG later
        self.SObs_0 = np.zeros(1)
        self.SObs_1 = np.zeros(1)
        self.SObs_2 = np.zeros(1)

    ################################################################################

    """
    setPopulationVectorEi
    Overload of setPopulationEi to set Ei directly from a vector, rather than a Pandas dataframe.
    """
    def setPopulationVectorEi(self, Ei):
        self.Ei = Ei
        assert len(self.Ei) == self.m, "FATAL: setPopulationEi length Ei=" + str(
            len(self.Ei)) + " MUST equal model definition size of m=" + str(self.m)

    ################################################################################

    """
    setPopulationEi
    Given a data frame containing one column with the zone number (i) and one column
    with the actual data values, fill the Ei population property with data so that
    the position in the Ei numpy array is the zonei field of the data.
    The code is almost identical to the setAttractorsAj method.
    NOTE: max(i) < self.m
    """
    def setPopulationEi(self, df, zoneiColName, dataColName):
        df2 = df.sort_values(by=[zoneiColName])
        self.Ei = df2[dataColName].to_numpy()
        assert len(self.Ei) == self.m, "FATAL: setPopulationEi length Ei=" + str(len(self.Ei)) + " MUST equal model definition size of m=" + str(self.m)

    ################################################################################

    """
    setAttractorsAj
    Given a data frame containing one column with the zone number (j) and one column
    with the actual data values, fill the Aj attractors property with data so that
    the position in the Aj numpy array is the zonej field of the data.
    The code is almost identical to the setPopulationEi method.
    NOTE: max(j) < self.n
    """
    def setAttractorsAj(self, df, zonejColName, dataColName):
        df2 = df.sort_values(by=[zonejColName])
        self.Aj = df2[dataColName].to_numpy()
        assert len(self.Aj) == self.n, "FATAL: setAttractorsAj length Aj=" + str(len(self.Aj)) + " MUST equal model definition size of n=" + str(self.n)

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
    setObsMatrix
    Assign the cost matrix for the model to use when it runs.
    NOTE: this MUST match the m x n order of the model and be a numpy array
    """
    def setObsMatrix(self, SObs_0, SObs_1, SObs_2):
        i, j = SObs_0.shape  # Hopefully cij_0.shape = SObs_0.shape = SObs_0.shape
        assert i == self.m, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.m=" + str(
            i) + " MUST match model definition of m=" + str(self.m)
        assert j == self.n, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.n=" + str(
            j) + " MUST match model definition of n=" + str(self.n)
        self.SObs_0 = SObs_0
        self.SObs_1 = SObs_1
        self.SObs_2 = SObs_2

    ################################################################################

    """
    computeCBar
    Compute average trip length TODO: VERY COMPUTATIONALLY INTENSIVE - FIX IT
    @param Sij trips matrix containing the flow numbers between MSOA (i) and schools (j)
    @param cij trip times between i and j
    """
    def computeCBar(self, Sij, cij):
        CNumerator = np.sum(Sij * cij)
        CDenominator = np.sum(Sij)
        cbar = CNumerator / CDenominator
        return cbar

    ################################################################################

    """
        Calculate Dj for a trips matrix.
        Two methods are presented here, one which is simple and very slow and one
        which uses python vector maths and is much faster. Once 2 is proven equal
        to 1, then it can be used exclusively. This function is mainly used for
        testing with the TensorFlow and other implementations.
        """

    def calculateDj(self, Tij):
        (M, N) = np.shape(Tij)
        Dj = np.zeros(N)
        Dj = Tij.sum(axis=0)
        return Dj

    ###############################################################################

    """
    run Model run3modes
    Quant model for three modes of transport
    @returns Sij predicted flows between i and j
    """
    def run3modes(self):
        # run model
        # i = employment zone
        # j = residential zone
        # Ei = number of jobs at MSOA location
        # Aj = attractor of HH (number of dwellings)
        # cij_mode = travel cost for "mode" (i.e. road, bus, rail)
        # Modes: Road = 0, Bus = 1, Rail = 2
        # Beta = Beta values for three modes - this is also output
        # QUANT data:
        # Observed trips data: "SObs_1.bin", "SObs_2.bin", "SObs_3.bin"
        # Travel cost per mode: "dis_roads_min.bin", "dis_bus_min.bin", "dis_gbrail_min.bin"
        # Note the use of 1,2,3 for modes in the files different from 0,1,2 in the code.
        # Returns predicted flows per mode: "SPred_1.bin", "SPred_2.bin", "SPred_3.bin"

        # Initialisation of parameters
        converged = False  # initialise convergence check param
        n_modes = 3  # Number of modes
        cij_k = [self.cij_0, self.cij_1, self.cij_2]  # list of cost matrices
        SObs_k = [self.SObs_0, self.SObs_1, self.SObs_2]  # list of obs trips matrices

        # Set up Beta for modes 0, 1 and 2 to 1.0
        Beta = np.ones(n_modes)

        # Compute sum of origins and destinations

        '''
        # OiObs : vector with dimension = number of oringins
        OiObs = np.zeros(self.n)
        for k in range(n_modes):
            OiObs += SObs_k[k].sum(axis=1)
        '''

        # DjObs : vector with dimension = number of destinations
        DjObs = [[] for i in range(n_modes)]
        for k in range(n_modes):
            DjObs[k] = np.zeros(self.n)
        for k in range(n_modes):
            DjObs[k] += SObs_k[k].sum(axis=1)

        DjPred = [[] for i in range(n_modes)]
        DjPredSum = np.zeros(n_modes)
        DjObsSum = np.zeros(n_modes)
        delta = np.zeros(n_modes)


        # Convergence loop:
        print("Calibrating the model...")
        iteration = 0
        while converged != True:
            iteration += 1
            print("Iteration: ", iteration)

            # Initialise variables:
            Sij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty lists equal to n_modes

            # hold copy of pre multiplied copies of -Beta_k * cij[k] for each mode
            ExpMBetaCijk = [[] for k in range(n_modes)]
            for kk in range(n_modes):
                ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

            for k in range(n_modes):  # mode loop
                Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
                for i in range(self.m):
                    denom = 0
                    for kk in range(n_modes):
                        denom += np.sum(self.Aj * ExpMBetaCijk[kk][i, :])
                    Sij2 = self.Ei[i] * (self.Aj * ExpMBetaCijk[k][i, :] / denom)
                    Sij[k][i, :] = Sij2  # put answer slice back in return array

            # Calibration with CBar values
            # Calculate mean predicted trips and mean observed trips (this is CBar)
            CBarPred = np.zeros(n_modes)
            CBarObs = np.zeros(n_modes)
            delta = np.ones(n_modes)
            for k in range(n_modes):
                CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])
                CBarObs[k] = self.computeCBar(SObs_k[k], cij_k[k])
                delta[k] = np.absolute(CBarPred[k] - CBarObs[k])  # the aim is to minimise delta[0]+delta[1]+...
            # delta check on all betas (Beta0, Beta1, Beta2) stopping condition for convergence
            # double gradient descent search on Beta0 and Beta1 and Beta2
            converged = True
            for k in range(n_modes):
                if (delta[k] / CBarObs[k] > 0.001):
                    Beta[k] = Beta[k] * CBarPred[k] / CBarObs[k]
                    converged = False
            '''
            # Calibration with Observed flows
            for k in range(n_modes):
                DjPred[k] = self.calculateDj(Sij[k])
                DjPredSum[k] = np.sum(DjPred[k])
                DjObsSum[k] = np.sum(DjObs[k])
                delta[k] = DjPredSum[k] - DjObsSum[k]
            # delta check on beta stopping condition for convergence
            # gradient descent search on beta
            converged = True
            for k in range(n_modes):
                if (delta[k] / DjObsSum[k] > 0.001):
                    Beta[k] = Beta[k] * DjPredSum[k] / DjObsSum[k]
                    converged = False
            '''
            CBarPred = np.zeros(n_modes)
            # Calculate CBar
            for k in range(n_modes):
                CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])

            # Debug:
            # commuter sum blocks
            TotalSij_roads = Sij[0].sum()
            TotalSij_bus = Sij[1].sum()
            TotalSij_rail = Sij[2].sum()
            TotalEi = self.Ei.sum()  # total jobs = pu+pr above
            # print("i= {0:d} beta_roads={1:.6f} beta_bus={2:.6f} beta_rail={3:.6f} cbar_pred_roads={4:.1f} cbar_pred_busr={5:.1f} cbar_pred_rail={6:.1f}"
            #         .format(iteration, Beta[0], Beta[1], Beta[2], CBarPred[0], CBarPred[1], CBarPred[2]))
            # print("TotalSij_roads={0:.1f} TotalSij_bus={1:.1f} TotalSij_rail={2:.1f} Total={3:.1f} ({4:.1f})"
            #       .format(TotalSij_roads, TotalSij_bus, TotalSij_rail, TotalSij_roads + TotalSij_bus + TotalSij_rail, TotalEi))

        return Sij, Beta, CBarPred  # Note that Sij = [Sij_k=0 , Sij_k=1, Sij_k=2] and CBarPred = [CBarPred_0, CBarPred_1, CBarPred_2]

    ################################################################################

    """
    run Model run3modes_NoCalibration
    Quant model for three modes of transport without calibration
    @returns Sij predicted flows between i and j
    """
    def run3modes_NoCalibration(self, Beta):
        n_modes = len(Beta)  # Number of modes
        print("Running model for ", n_modes, " modes.")

        # Initialise variables:
        Sij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty lists equal to n_modes
        cij_k = [self.cij_0, self.cij_1, self.cij_2]  # list of cost matrices

        # hold copy of pre multiplied copies of -Beta_k * cij[k] for each mode
        ExpMBetaCijk = [[] for k in range(n_modes)]
        for kk in range(n_modes):
            ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

        for k in range(n_modes):  # mode loop
            Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
            for i in range(self.m):
                denom = 0
                for kk in range(n_modes):
                    denom += np.sum(self.Aj * ExpMBetaCijk[kk][i, :])
                Sij2 = self.Ei[i] * (self.Aj * ExpMBetaCijk[k][i, :] / denom)
                Sij[k][i, :] = Sij2  # put answer slice back in return array

        CBarPred = np.zeros(n_modes)  # initialise CBarPred
        for k in range(n_modes):
            CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])

        return Sij, CBarPred

    ################################################################################

    """
    computeProbabilities3modes
    Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones
    """
    def computeProbabilities3modes(self, Sij):
        print("Computing probabilities")
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