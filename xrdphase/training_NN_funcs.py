"""This module contains the functions to create the training set of a neural
network for PhaseIdentification.

"""

import pickle
import random
import numpy as np
from sklearn.neural_network import MLPClassifier


def convert_tth_to_q(tth, lam=1.5406):
    """Converts 2theta values to Q values.

    Needed so that the sample data and models use the same units. 2theta is
    energy dependent, so we need to convert the 2theta values to Q, where
    energy is not required.

    Parameters
    ----------
    `tth` : float
            2theta values of the models from the Materials Project
    `lam` : float
            Wavelength in angstroms, default value is Cu K-alpha value

    """

    q = (np.pi*4.0/lam)*np.sin(tth/360*np.pi)
    return q


def gauss_func(x, wid, cen, amp):
    """Single Gaussian curve

    Used to create the training set for the neural network

    Parameters
    ----------
    `x` : array_like
          X values to be used in the function
    `wid` : float
            Width of the peak
    `cen` : float
            Center of the peak
    `amp` : float
            Amplitude of the peak

    """

    return np.exp(-((x-cen)**2.)/(2.*wid**2)) * amp


def trainNN():
    """Creates the training set and trains the neural network on it

    This neural network identifies whether or not a peak exists within a
    certain range. The training set consists of full peaks (label=1), peaks on
    the edges of the chosen range (label=0), and 0-slope lines (label=0).

    Returns
    -------
    `clf` : MLPClassifier
            This is the trained neural network ready to use

    Notes
    -----
    Hidden layer size: 1 layer of 100 neurons (default)
    Activation function: relu (default)
    Weight optimization solver: LBFGS
    L2 penalty (regularization term): 0.0001 (default)

    """

    yTrain = []  # holds y vals of curves/lines
    trainLabels = []  # holds center labels

    tryCenters = np.linspace(1, 9, 45)

    for i in range(len(tryCenters)):
        x = np.linspace(tryCenters[i]-.2, tryCenters[i]+.2, 18)
        for j in range(1000):
            centers = round(random.uniform(tryCenters[i]-.05,
                                           tryCenters[i]+.05), 1)
            y = gauss_func(x, .05, centers, 1)
            yTrain.append(y)
            trainLabels.append(1)

            y = gauss_func(x, .05,
                           round(random.uniform(tryCenters[i]-.3,
                                                tryCenters[i]-.17), 1), 1)
            yTrain.append(y)
            trainLabels.append(0)

            y = gauss_func(x, .05,
                           round(random.uniform(tryCenters[i]+.17,
                                                tryCenters[i]+.3), 1), 1)
            yTrain.append(y)
            trainLabels.append(0)

            y = 0*x
            yTrain.append(y)
            trainLabels.append(0)
    clf = MLPClassifier(solver='lbfgs')
    clf.fit(yTrain, trainLabels)
    return clf


def pickle_nn(clf):
    """Pickles/serializes the trained neural network so it can be used later.

    This enables the network to be used without having to create it repeatedly

    Parameters
    ----------
    `clf` : MLPClassifier
            This is the trained neural network

    """

    filename = 'nnMLPClass'
    outfile = open(filename, 'wb')
    pickle.dump(clf, outfile)
    outfile.close()
