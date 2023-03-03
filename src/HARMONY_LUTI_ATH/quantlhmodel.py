"""
QUANTLHModel.py

Lakshmanan and Hansen form model. Used as a base for other models.
Uses an attractor, Aj, Population Ei and cost matrix Cij.
"""

import numpy as np
import pandas as pd

from HARMONY_LUTI_ATH.globals import *


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
        self.cij_pu = np.zeros(1)  # costs matrix - set to something BIG later
        self.cij_pr = np.zeros(1)  # costs matrix - set to something BIG later
        self.OD_pu = np.zeros(1)
        self.OD_pr = np.zeros(1)
        self.OBS_cbar_pu = 0
        self.OBS_cbar_pr = 0

    # end def constructor

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

    def setCostMatrixCij(self, cij_pu, cij_pr):
        i, j = cij_pu.shape  # Hopefully cij_pr.shape = cij_pu.shape
        assert i == self.m, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.m=" + str(
            i) + " MUST match model definition of m=" + str(self.m)
        assert j == self.n, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.n=" + str(
            j) + " MUST match model definition of n=" + str(self.n)
        self.cij_pu = cij_pu
        self.cij_pr = cij_pr

    ################################################################################

    """
    setObsMatrix
    Assign the cost matrix for the model to use when it runs.
    NOTE: this MUST match the m x n order of the model and be a numpy array
    """

    def setODMatrix(self, OD_pu, OD_pr):
        i, j = OD_pu.shape  # Hopefully cij_pr.shape = OD_0.shape
        assert i == self.m, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.m=" + str(
            i) + " MUST match model definition of m=" + str(self.m)
        assert j == self.n, "FATAL: setCostsMatrix cij matrix is the wrong size, cij.n=" + str(
            j) + " MUST match model definition of n=" + str(self.n)
        self.OD_pu = OD_pu
        self.OD_pr = OD_pr

    ################################################################################

    """
        setObsCbar
        Assign the observed cbar values for model calibration.
    """

    def setObsCbar(self, OBS_cbar_pu, OBS_cbar_pr):
        self.OBS_cbar_pu = OBS_cbar_pu
        self.OBS_cbar_pr = OBS_cbar_pr

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
    run Model run2modes
    Lakshmanan Hansen Model Two Modes
    """
    def runTwoModes(self, logger):
        # run model
        # i = employment zone
        # j = residential zone
        # Ei = number of jobs per zone
        # Aj = attractor of HH (floor space)
        # cij_mode = travel cost for "mode"
        # Modes: public and private transportation
        # Beta = Beta values for 2 modes - this is also output
        # Observed cbar values: OBS_cbar_pu, OBS_cbar_pr
        # Travel cost per mode: cij_pu, cij_pr

        # Initialisation of parameters
        n_modes = 2  # Number of modes
        cij_k = [self.cij_pu, self.cij_pr]  # list of cost matrices

        CBarObs = [self.OBS_cbar_pu, self.OBS_cbar_pr]

        # Set up Beta for modes 0, 1 to 1.0 as a starting point
        Beta = np.ones(n_modes)

        logger.warning("Calibrating the model...")
        iteration = 0
        converged = False
        while not converged:
            iteration += 1
            # print("Iteration: ", iteration)

            # Initialise variables:
            Sij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty lists equal to n_modes

            # hold copy of pre multiplied copies of -Beta_k * cij[k] for each mode
            ExpMBetaCijk = [[] for k in range(n_modes)]
            for kk in range(n_modes):
                ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

                # this is the main model loop to calculate Sij[k] trip numbers for each mode k
            for k in range(n_modes):  # mode loop
                Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
                for i in range(self.m):
                    denom = 0
                    for kk in range(n_modes):
                        denom += np.sum(self.Aj * ExpMBetaCijk[kk][i, :])
                    Sij2 = self.Ei[i] * (self.Aj * ExpMBetaCijk[k][i, :] / denom)
                    Sij[k][i, :] = Sij2  # put answer slice back in return array
            # end for k

            # now we see how well it worked and modify the two betas accordingly
            # we have no Oi or Dj to check against, so it can only be CBar and totals

            # Calculate mean predicted trips and mean observed trips (this is CBar)
            converged = True
            CBarPred = np.zeros(n_modes)
            delta = np.ones(n_modes)
            for k in range(n_modes):
                CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])
                delta[k] = np.absolute(CBarPred[k] - CBarObs[k])  # the aim is to minimise delta[0]+delta[1]+...
                # beta mode here and convergence check
                if delta[k] / CBarObs[k] > 0.001:
                    Beta[k] = Beta[k] * CBarPred[k] / CBarObs[k]
                    converged = False
            # end for k

            # commuter sum blocks
            TotalSij_pu = Sij[0].sum()
            TotalSij_pr = Sij[1].sum()
            TotalEi = self.Ei.sum()  # total jobs = pu+pr above

            # Debug block
            # for k in range(0,n_modes):
            #    print("Beta", k, "=", Beta[k])
            #    print("delta", k, "=", delta[k])
            # end for k
            # print("delta", delta[0], delta[1]) #should be a k loop
            logger.warning("iteration= {0:d} beta pu={1:.6f} beta pr={2:.6f} cbar pred pu={3:.1f} ({4:.1f}) cbar pred pr={5:.1f} ({6:.1f})"
                .format(iteration, Beta[0], Beta[1], CBarPred[0], CBarObs[0], CBarPred[1], CBarObs[1]))
            # print("TotalSij_pu={0:.1f} TotalSij_pr={1:.1f} Total={2:.1f} ({3:.1f})"
            #       .format(TotalSij_pu, TotalSij_pr, TotalSij_pu + TotalSij_pr, TotalEi))

        # end while ! converged

        return Sij, Beta, CBarPred, CBarObs  # Note that Sij = [Sij_k=0 , Sij_k=1] and CBarPred = [CBarPred_0, CBarPred_1]

    # end rwm_run2modes

        ################################################################################

    """
    run Model run2modes_NoCalibration
    Quant model for two modes of transport without calibration
    @returns Sij predicted flows between i and j
    """
    def run2modes_NoCalibration(self, Beta):
        n_modes = len(Beta)  # Number of modes
        cij_k = [self.cij_pu, self.cij_pr]  # list of cost matrices
        print("Running model for ", n_modes, " modes.")
        # Initialise variables:
        # Initialise variables:
        Sij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty lists equal to n_modes

        # hold copy of pre multiplied copies of -Beta_k * cij[k] for each mode
        ExpMBetaCijk = [[] for k in range(n_modes)]
        for kk in range(n_modes):
            ExpMBetaCijk[kk] = np.exp(-Beta[kk] * cij_k[kk])

        # this is the main model loop to calculate Sij[k] trip numbers for each mode k
        for k in range(n_modes):  # mode loop
            Sij[k] = np.zeros(self.m * self.n).reshape(self.m, self.n)
            for i in range(self.m):
                denom = 0
                for kk in range(n_modes):
                    denom += np.sum(self.Aj * ExpMBetaCijk[kk][i, :])
                Sij2 = self.Ei[i] * (self.Aj * ExpMBetaCijk[k][i, :] / denom)
                Sij[k][i, :] = Sij2  # put answer slice back in return array

        CBarPred = np.zeros(n_modes)
        for k in range(n_modes):
            CBarPred[k] = self.computeCBar(Sij[k], cij_k[k])

        return Sij, CBarPred

    ################################################################################

    """
    computeProbabilities2modes
    Compute the probability of a flow from an MSOA zone to any (i.e. all) of the possible point zones
    """

    def computeProbabilities2modes(self, Sij):
        n_modes = 2
        probSij = [[] for i in range(n_modes)]  # initialise Sij with a number of empty list equal to n_modes

        for k in range(n_modes):
            probSij[k] = np.arange(self.m * self.n, dtype=np.float).reshape(self.m, self.n)
            for i in range(self.m):
                sum = np.sum(Sij[k][i,])
                if sum <= 0:
                    sum = 1  # catch for divide by zero - just let the zero probs come through to the final matrix
                probSij[k][i,] = Sij[k][i,] / sum

        return probSij

    ################################################################################

    @staticmethod
    def loadODData():
        # load observed trips for each mode:

        # Import the csv OD public as Pandas DataFrame:
        OD_pu_df = pd.read_csv(os.path.join(datadir_ex, ODPuFilename_csv), header=None)
        # Convert the dataframe to a nupmy matrix:
        OD_pu = OD_pu_df.to_numpy()

        # Import the csv OD private as Pandas DataFrame:
        OD_pr_df = pd.read_csv(os.path.join(datadir_ex, ODPrFilename_csv), header=None)
        # Convert the dataframe to a nupmy matrix:
        OD_pr = OD_pr_df.to_numpy()

        return OD_pu, OD_pr

    ################################################################################

