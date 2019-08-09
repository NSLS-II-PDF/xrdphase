"""This module contains functions used in phase_id_script to find the most
similar crystal phase of a material using X-ray diffraction at a beamline.

"""

import numpy as np
import matplotlib.pyplot as plt
from pymatgen import MPRester
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pickle


def get_structures(apiKey, elements):
    """Gets structures from Materials Project database.

    Uses a Materials Project API key to query the database with a list of
    elements. Gets Cu K-alpha diffraction pattern, material id, and spacegroup
    info for each model.

    Parameters
    ----------
    `apiKey` : str
               This is your API key from the MaterialsProject.
    `elements` : list of str
                 Include a list of elements in the sample here.

    Returns
    -------
    `models` : list of dicts
               Contains X-ray diffraction patterns, material ids, and space
               group information about the elements queried.

    """

    mpr = MPRester(apiKey)
    numElements = len(elements)
    models = mpr.query(criteria={"elements": {"$all": [*elements]},
                                 "nelements": numElements},
                       properties=['xrd.Cu', 'material_id', 'spacegroup'])
    return models


def read_data(fileName, startCut):
    """Reads sample data file and cuts data.

    Reads the data file and makes an initial cut at the Qmin value the user
    specifies. Qmax is set to 20.

    Parameters
    ----------
    `fileName` : str
                 This is the name of the sample data file. This file needs to
                 contain Q values.
    `startCut` : int
                 Qmin. This is the value which the data will start at after
                 being cut.

    Returns
    -------
    `qcut` : ndarray
             This is an array of Q values after the sample data has been cut.
             The range of Q values is from startCut to 20.
    `iqcut` : ndarray
              This is an array of I of Q values after the sample data has been
              cut.
    `numPeak` : int
                This is a broad number of peaks used to later limit which
                models are checked for peaks with the neural network.

    """

    q, iq = read_index_data_smart(fileName)
    qcut, iqcut = cut_data(q, iq, startCut, 20)
    iqcutN = iqcut/max(iqcut)
    peaks, _ = find_peaks(iqcutN, prominence=0.02)
    numPeaks = len(peaks)
    return qcut, iqcut, numPeaks


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def cut_data(qt, sqt, qmin, qmax):
    qt_back, sqt_back = qt[qt > qmin], sqt[qt > qmin]
    qt_back, sqt_back = qt_back[qt_back < qmax], sqt_back[qt_back < qmax]
    return qt_back, sqt_back


def read_index_data_smart(filename, junk=None, backjunk=None, splitchar=None,
                          do_not_float=False, shh=True, use_idex=[0, 1]):
    with open(filename, 'r') as infile:
        datain = infile.readlines()
    if junk is None:
        for i in range(len(datain)):
            try:
                for j in range(10):
                    float(datain[i+j].split(splitchar)[use_idex[0]])
                    float(datain[i+j].split(splitchar)[use_idex[1]])
                junk = i
                break
            except Exception:
                pass

    if backjunk is None:
        for i in range(len(datain), -1, -1):
            try:
                float(datain[i].split(splitchar)[use_idex[0]])
                float(datain[i].split(splitchar)[use_idex[1]])
                backjunk = len(datain)-i-1
                break
            except Exception:
                pass
    if backjunk == 0:
        datain = datain[junk:]
    else:
        datain = datain[junk:-backjunk]
    xin = np.zeros(len(datain))
    yin = np.zeros(len(datain))
    if do_not_float:
        xin = []
        yin = []
    if shh is False:
        print('length '+str(len(xin)))
    if do_not_float:
        if splitchar is None:
            for i in range(len(datain)):
                xin.append(datain[i].split()[use_idex[0]])
                yin.append(datain[i].split()[use_idex[1]])
        else:
            for i in range(len(datain)):
                xin.append(datain[i].split(splitchar)[use_idex[0]])
                yin.append(datain[i].split(splitchar)[use_idex[1]])
    else:
        if splitchar is None:
            for i in range(len(datain)):
                xin[i] = float(datain[i].split()[use_idex[0]])
                yin[i] = float(datain[i].split()[use_idex[1]])
        else:
            for i in range(len(datain)):
                xin[i] = float(datain[i].split(splitchar)[use_idex[0]])
                yin[i] = float(datain[i].split(splitchar)[use_idex[1]])

    return xin, yin


def bgd_func(qcut, iqcutStart, iqcutEnd):
    """Linear background function

    Equation of a line used to eliminate the background of the sample data.

    Parameters
    ----------
    `qcut` : ndrray
             Q values of the sample data after it's been cut. First and last
             values are used to determine slope.
    `iqcutStart` : float
                   The first I(Q) value of the data after it has been cut again
                   based on the model from the Materials Project.
    `iqcutEnd` : float
                 The last I(Q) value of the data after it has been cut again
                 based on the model from the Materials Project.

    Notes
    -----
    This function may be modified to use a different background function.
    A higher order polynomial may provide a better fit than a line.

    """

    a1 = (iqcutEnd-iqcutStart)/(qcut[-1]-qcut[0])
    a0 = iqcutStart-(qcut[0]*(a1))
    return a0 + a1*qcut


def sum_gauss(x, *params):
    """Sum of gaussians.

    Used to fit a curve to the sample data based on an initial guess and bounds
    from the models. Needed to find residuals for each model to determine the
    closest fit model.

    Parameters
    ----------
    `x` : array_like
          X values to be used in the funciton.
    `*params` : list of float
                list containing centers, widths, and amplitudes for each peak

    """

    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        cen = params[i]
        wid = params[i+1]
        amp = params[i+2]
        y = y + np.exp(-((x-cen)**2)/(2.*wid**2))*amp
    return y


def convert_tth_to_q(tth, lam=1.5406):
    """Converts 2theta values to Q values

    Needed so that sample data and models use the same units. 2theta is energy
    dependent, so we need to convert the 2theta values to Q, where energy is
    not required.

    Parameters
    ----------
    `tth` : float
            2theta values of the models from the Materials Project
    `lam` : float
            Wavelength in angstroms, default value is Cu K-alpha value

    """

    q = (np.pi*4.0/lam)*np.sin(tth/360*np.pi)
    return q


def cut_data_length(qt, sqt, qmin, length):
    """Cuts the data while maintaining the length passed in.

    Ensures the data starts at qmin and ends when the length has been achieved.

    Parameters
    ----------
    `qt` : array_like
           Q values to be cut
    `sqt` : array_like
            I(Q) values to be cut
    `qmin` : float
             Minimum Q value after data is cut
    `length` : int
               Number of elements required after the data has been cut

    Returns
    -------
    `qcut` : ndarray
             The array of Q values after the sample data has been cut.
    `sqcut` : ndarray
              The array of I of Q values after the sample data has been cut.

    """

    qcut = []
    sqcut = []
    for i in range(len(qt)):
        if qt[i] >= qmin and len(qcut) < length:
            qcut.append(qt[i])
            sqcut.append(sqt[i])
    qcut = np.array(qcut)
    sqcut = np.array(sqcut)
    return qcut, sqcut


def identify_phase(models, qcut, iqcut, q_dep_shift=1.0, const_shift=0.0):
    """Identifies the most likely material id and space group symbol from the
    Materials Project.

    Applies a Q-dependent or constant shift and tries to fit a curve to the
    sample data based on where the peaks are in the models from the database.
    There's a penalty for the sample data having a peak where the models don't,
    and another for the models having peaks where the data doesn't.
    Then the the model is picked where the residual is the lowest.

    Parameters
    ----------
    `models` : list of dicts
               The list containing the X-ray diffraction patterns, material
               ids, and space group symbols from the query to the Materials
               Project.
    `qcut` : ndarray
             Q values from the cut sample data
    `iqcut` : ndarray
              I(Q) values from the cut sample data
    `const_shift` : float, optional
                    This adds the shift amount to every model Q value.
                    Default=0.0
    `q_dep_shift` : float, optional
                    This multiplies the shift amount to every Q value.
                    Default=1.0


    Returns
    -------
    `fitIndx` : int
                Index of the model picked as the correct model in the list of
                models from the query to the database

    Notes
    -----
    This function first gets the 2theta and amplitude values for each model
    from the database and converts the 2theta values to Q. Then it cuts the
    sample data again based on where each model ends, and it subtracts the
    background out of the data (using the first and last values of the newly
    cut data). It finds the number of peaks in the data to eliminate models
    later on, and then it creates and stores initial guesses and bounds for
    curve fitting. It cuts out models with too many peaks, then tries to fit a
    curve to the data and stores the residuals. It goes through the list of
    residuals and prints the most likely and then second and third most likely,
    if they exist, models with their material ids and space group symbols.

    """

    resSumList = []  # will hold the residuals for each model
    modelList = []  # holds index of models that will have graphs
    for j in range(len(models)):
        model_num = models[j]['xrd.Cu']['pattern']
        x_val_mod = []
        y_val_mod = []
        effective_peak_x = []
        effective_peak_y = []

        for i in range(len(model_num)):
            x_val_mod.append(convert_tth_to_q(model_num[i][2])*.995*q_dep_shift
                             + const_shift)
            y_val_mod.append(model_num[i][0])
            # choosing the peaks to use based on where the data starts
            if x_val_mod[i] >= qcut[0]:
                effective_peak_x.append(x_val_mod[i])
                effective_peak_y.append(y_val_mod[i])

        # cut the data again for each model
        qcutM, iqcutM = cut_data(qcut, iqcut, 0, effective_peak_x[-1]+.1)

        bgd = bgd_func(qcutM, iqcutM[0], iqcutM[-1])

        iqcutNoBgd = np.zeros_like(iqcutM)
        iqcutNoBgd = iqcutM-bgd
        for i in range(len(iqcutNoBgd)):
            if iqcutNoBgd[i] < 0:
                iqcutNoBgd[i] = 0

        # find the number of sample data peaks
        peaks, _ = find_peaks(iqcutNoBgd, height=10, prominence=1)

        initial_guess = []
        low_lims = []
        high_lims = []
        zero_count = 0
        residual = 0
        for i in range(len(effective_peak_x)):
            if effective_peak_x[i] >= qcut[2]:
                x = effective_peak_x[i]
                # used to see if there's a better peak somewhere else
                peak_max_indx = 2
                indx = find_nearest(qcut, x)

                my_max = np.max(iqcutNoBgd[indx-peak_max_indx:
                                indx+peak_max_indx])
                my_max_indx = find_nearest(iqcutNoBgd, my_max)

                initial_guess.append(qcut[my_max_indx])  # cen
                initial_guess.append(.04)  # wid
                initial_guess.append(iqcutNoBgd[my_max_indx])  # amp
                if iqcutNoBgd[my_max_indx] == 0:
                    # keeping a count of all the areas where amplitude is 0
                    zero_count += 1

                low_lims.append(qcut[my_max_indx]-.03)  # cen
                low_lims.append(.0002)  # wid
                low_lims.append(0)  # amp

                high_lims.append(qcut[my_max_indx]+.03)  # cen
                high_lims.append(.06)  # wid
                high_lims.append(max(iqcutNoBgd))  # amp

        # want to show progress
        print("Working on model " + str(j+1) + " of " + str(len(models)))

        # cuts out models with way too many peaks
        if len(effective_peak_x) <= 2*len(qcutM[peaks]):
            try:
                fit_data, _ = curve_fit(sum_gauss, qcutM, iqcutNoBgd,
                                        p0=initial_guess,
                                        bounds=([low_lims, high_lims]),
                                        maxfev=100)
                print("A curve was fit for " + models[j]['material_id'])

                residual += abs(iqcutNoBgd-sum_gauss(qcutM, *fit_data))
                sumRes = (sum(residual)/len(effective_peak_x))+zero_count*1000
                resSumList.append(sumRes)
                modelList.append(j)

            except RuntimeError:
                residual = 100000+zero_count*1000
                print("Curve fit failed for " + models[j]['material_id'])
        else:
            print("I think curve fit will fail for " +
                  models[j]['material_id'] +
                  ". There might be too many peaks")

    # zip resSumList and modelList together to rank residuals
    mappedRes = list(zip(resSumList, modelList))

    # sort mapped residual values
    mappedRes.sort()

    # print top choices
    if len(modelList) == 0:
        print("\nSorry, no matches were found at all")
        fitIndx = None
        print()
        return fitIndx

    elif len(modelList) == 1:
        fitIndx = mappedRes[0][1]
        print("\nI think you're looking at " +
              models[fitIndx]['spacegroup']['symbol'] + ", " +
              models[fitIndx]['material_id'])
        print()
        return fitIndx

    elif len(modelList) == 2:
        fitIndx = mappedRes[0][1]
        print("\nI think you're looking at " +
              models[fitIndx]['spacegroup']['symbol'] + ", " +
              models[fitIndx]['material_id'])
        # print runner up
        print("Model " + models[mappedRes[1][1]]['material_id'] + ", " +
              models[mappedRes[1][1]]['spacegroup']['symbol'] +
              " has the second smallest residual.")
        print()
        return fitIndx

    else:
        fitIndx = mappedRes[0][1]
        print("\nI think you're looking at " +
              models[fitIndx]['spacegroup']['symbol'] + ", " +
              models[fitIndx]['material_id'])
        # print runner ups
        print("Model " + models[mappedRes[1][1]]['material_id'] + ", " +
              models[mappedRes[1][1]]['spacegroup']['symbol'] +
              " has the second smallest residual.")
        print("Model " + models[mappedRes[2][1]]['material_id'] + ", " +
              models[mappedRes[2][1]]['spacegroup']['symbol'] +
              " has the third smallest residual.")
        print()
        return fitIndx


def get_NN():
    """This unpickles the MLPClassifier neural network.

    The neural network should already be made, so this calls the NN to use.

    Returns
    -------
    `clf` : MLPClassifier
            This is the trained neural network used to identify if a peak
            exists in a range or not.

    """

    infile = open('nnMLPClass', 'rb')
    clf = pickle.load(infile)
    infile.close()
    return clf


def identify_phase_nn(models, qcut, iqcut, clf, numPeaks, q_dep_shift=1.0,
                      const_shift=0.0):
    """Identifies the most likely material id and space group symbol from the
    Materials Project using a neural network.

    This goes through each model and cuts out sections from the sample data at
    the positions that each model has a peak. A constant or Q-dependent shift
    can be applied to fix calculated MP peak positions. It then predicts if a
    peak existsin that section by using a neural network trained on peaks that
    exist andpeaks that don't exist (lines with 0 slope, curves on the edges).
    It picks out the model with the most peak matches as the most likely model.

    Parameters
    ----------
    `models` : list of dicts
               The list containing the X-ray diffraction patterns, material
               ids, and space group symbols from the query to the Materials
               Project.
    `qcut` : ndarray
             Q values from the cut sample data
    `iqcut` : ndarray
              I(Q) values from the cut sample data
    `clf` : MLPClassifier
            This is the trained classifier neural network that identifies if a
            peak exists within a range
    `numPeaks` : int
                 This is the broad number of peaks used to cut out models with
                 very high numbers of peaks from being tested with the NN
    `const_shift` : float, optional
                    This adds the shift amount to every model Q value.
                    Default=0.0
    `q_dep_shift` : float, optional
                    This multiplies the shift amount to every Q value.
                    Default=1.01
    Returns
    -------
    `bestMatch` : int
                  Index of the model picked as the closest match in the list of
                  models from the query to the database

    Notes
    -----
    This function does not return a top three ranking like the other identify
    phase function, but it seems to perform faster.

    """

    pred = []  # holds prediction to do some more analysis later
    predModelNum = []
    for j in range(len(models)):
        model_num = models[j]['xrd.Cu']['pattern']
        x_val_mod = []
        y_val_mod = []
        effective_peak_x = []
        effective_peak_y = []
        yTest = []

        # read in model peak positions, convert to Q
        for i in range(len(model_num)):
            x_val_mod.append(convert_tth_to_q(model_num[i][2])*.995*q_dep_shift
                             + const_shift)
            y_val_mod.append(model_num[i][0])

            if x_val_mod[i] >= qcut[0]:
                effective_peak_x.append(x_val_mod[i])
                effective_peak_y.append(y_val_mod[i])

        # cut data for each peak
        if len(x_val_mod) <= 2*numPeaks:

            for i in range(len(effective_peak_x)):
                qcutM, iqcutM = cut_data_length(qcut, iqcut,
                                                effective_peak_x[i]-.2, 18)
                bgd = bgd_func(qcutM, iqcutM[0], iqcutM[-1])

                iqcutNoBgd = np.zeros_like(iqcutM)
                iqcutNoBgd = iqcutM-bgd
                for i in range(len(iqcutNoBgd)):
                    if iqcutNoBgd[i] < 0:
                        iqcutNoBgd[i] = 0

                if max(iqcutNoBgd) == 0:
                    iqcutNoBgd = np.zeros_like(iqcutNoBgd)

                else:
                    iqcutNoBgd = iqcutNoBgd/max(iqcutNoBgd)
                yTest.append(iqcutNoBgd)

            pred.append(clf.predict(yTest))
            predModelNum.append(j)

    # find the best match percentage and return that model index
    fitBool = []
    matchPercentList = []
    for i in range(len(pred)):
        totalMatches = len(pred[i])
        goodMatches = sum(pred[i])
        badMatches = len(pred[i])-sum(pred[i])
        matchPercent = goodMatches/totalMatches*100
        matchPercentList.append(matchPercent)
        if badMatches >= goodMatches:
            fitBool.append(0)
        else:
            fitBool.append(1)

    bestMatch = predModelNum[matchPercentList.index(max(matchPercentList))]
    print("I think you're looking at " +
          models[bestMatch]['spacegroup']['symbol'] + ", " +
          models[bestMatch]['material_id'])
    return bestMatch


def show_correct_model(models, fitIndx, qcut, iqcut, const_shift=0.0,
                       q_dep_shift=1.0):
    """Graphs the correct model

    Uses the index returned from the identify_phase function and graphs the
    corresponding model with the data.

    Parameters
    ----------
    `models` : list of dicts
               The list containing the X-ray diffraction patterns, material
               ids, and space group symbols from the query to the Materials
               Project.
    `fitIndx` : int
                Index of the model in the list of models that was determined to
                be the correctly fit model
    `qcut` : ndarray
             Q values from the cut sample data
    `iqcut` : ndarray
              I(Q) values from the cut sample data
    `const_shift` : float, optional
                    This adds the shift amount to every model Q value.
                    Default=0.0
    `q_dep_shift` : float, optional
                    This multiplies the shift amount to every Q value.
                    Default=1.01

    Notes
    -----
    Since a number of MP model peak positions are calculated, some models
    exhibit a slight shift at higher Q values. The two shift parameters can be
    used to show what a correction would look like. A 1.01 q_dep_shift appears
    to be a good Q dependent shift value to correct the peak positions.

    """

    x_val_mod = []
    y_val_mod = []
    effective_peak_x = []
    effective_peak_y = []

    model_num = models[fitIndx]['xrd.Cu']['pattern']

    for i in range(len(model_num)):
        x_val_mod.append(convert_tth_to_q(model_num[i][2])*.995*q_dep_shift +
                         const_shift)
        y_val_mod.append(model_num[i][0])
        if x_val_mod[i] >= qcut[0]:
            effective_peak_x.append(x_val_mod[i])
            effective_peak_y.append(y_val_mod[i])

        qcutM, iqcutM = cut_data(qcut, iqcut, 0, effective_peak_x[-1]+.1)

        bgd = bgd_func(qcutM, iqcutM[0], iqcutM[-1])

        iqcutNoBgd = np.zeros_like(iqcutM)
        iqcutNoBgd = iqcutM-bgd
        for j in range(len(iqcutNoBgd)):
            if iqcutNoBgd[j] < 0:
                iqcutNoBgd[j] = 0

        initial_guess = []
        low_lims = []
        high_lims = []
        zero_count = 0
        residual = 0
        for j in range(len(effective_peak_x)):
            if effective_peak_x[j] >= qcut[2]:
                x = effective_peak_x[j]
                # used to see if there's a better peak somewhere else
                peak_max_indx = 2
                indx = find_nearest(qcut, x)

                my_max = np.max(iqcutNoBgd[indx-peak_max_indx:
                                indx+peak_max_indx])
                my_max_indx = find_nearest(iqcutNoBgd, my_max)

                initial_guess.append(qcut[my_max_indx])  # cen
                initial_guess.append(.04)  # wid
                initial_guess.append(iqcutNoBgd[my_max_indx])  # amp
                if iqcutNoBgd[my_max_indx] == 0:
                    # keeping a count of all the areas where amplitude is 0
                    zero_count += 1

                low_lims.append(qcut[my_max_indx]-.03)  # cen
                low_lims.append(.0002)  # wid
                low_lims.append(0)  # amp

                high_lims.append(qcut[my_max_indx]+.03)  # cen
                high_lims.append(.06)  # wid
                high_lims.append(max(iqcutNoBgd))  # amp

    try:
        fit_data, _ = curve_fit(sum_gauss, qcutM, iqcutNoBgd,
                                p0=initial_guess,
                                bounds=([low_lims, high_lims]),
                                maxfev=100)
        plt.figure()
        plt.title(models[fitIndx]['material_id'] + "  " +
                  models[fitIndx]['spacegroup']['symbol'])
        plt.plot(qcutM, iqcutNoBgd, label="Data")
        plt.plot(qcutM, sum_gauss(qcutM, *fit_data), label="Fit")
        plt.vlines(effective_peak_x, 0, initial_guess[2::3], label="Model",
                   colors='r')
        plt.xlabel("Q")
        plt.ylabel("Intensity")
        plt.legend(loc=0)

        residual += abs(iqcutNoBgd-sum_gauss(qcutM, *fit_data))
        sumRes = (sum(residual)/len(effective_peak_x))+zero_count*1000
        print("Sum of residual for " + models[fitIndx]['material_id'] + ": "
              + str(sumRes))

        plt.show()

    except RuntimeError:
        residual = 100000+zero_count*1000
        print("There weren't any fits")
