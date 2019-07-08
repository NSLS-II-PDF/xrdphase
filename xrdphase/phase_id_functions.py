"""This module contains functions used in phase_id_script to find the most
similar crystal phase of a material using X-ray diffraction at a beamline.

"""

import numpy as np
import matplotlib.pyplot as plt
from pymatgen import MPRester
from xrdphase.free_pdf import find_nearest, cut_data, read_index_data_smart
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


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

    Reads the data file and makes an initial cut at the start of the data
    based on the q value that the user specifies. The end of the data is cut at
    Q=20.

    Parameters
    ----------
    `fileName` : str
                 This is the name of the sample data file. This file needs to
                 contain Q values.
    `startCut` : int
                 This is the value which the data will start at after being
                 cut. Pick a value that cuts out air scatter and the beam stop.

    Returns
    -------
    `qcut` : ndarray
             This is an array of Q values after the sample data has been cut.
             The range of Q values is from startCut to 20.
    `iqcut` : ndarray
              This is an array of I of Q values after the sample data has been
              cut.

    """

    q, iq = read_index_data_smart(fileName)
    qcut, iqcut = cut_data(q, iq, startCut, 20)
    return qcut, iqcut


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
    dependent, so we need to convert the 2theta values to q, where energy is
    not required.

    Parameters
    ----------
    `tth` : float
            2theta values of the models from the Materials Project
    `lam` : float
            Wavelength in angstroms, default value is Cu K-alpha value

    """

    "Converts 2theta values in Materials Project models to q"
    q = (np.pi*4.0/lam)*np.sin(tth/360*np.pi)
    return q


def identify_phase(models, qcut, iqcut):
    """Identifies the most likely material id and space group symbol from the
    Materials Project.

    Tries to fit a curve to the sample data based on where the peaks are in the
    models from the database. There's a penalty for the sample data having a
    peak where the models don't, and another for the models having peaks where
    the data doesn't. Then the the model is picked where the residual is the
    lowest.

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
            x_val_mod.append(convert_tth_to_q(model_num[i][2])*.995)
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


def show_correct_model(models, fitIndx, qcut, iqcut):
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

    """

    x_val_mod = []
    y_val_mod = []
    effective_peak_x = []
    effective_peak_y = []

    model_num = models[fitIndx]['xrd.Cu']['pattern']

    for i in range(len(model_num)):
        x_val_mod.append(convert_tth_to_q(model_num[i][2])*.995)
        y_val_mod.append(model_num[i][0])
        if x_val_mod[i] >= qcut[0]:
            effective_peak_x.append(x_val_mod[i])
            effective_peak_y.append(y_val_mod[i])

        qcutM, iqcutM = cut_data(qcut, iqcut, 0, effective_peak_x[-1]+.1)

        bgd = bgd_func(qcutM, iqcutM[0], iqcutM[-1])

        iqcutNoBgd = np.zeros_like(iqcutM)
        iqcutNoBgd = iqcutM-bgd
        for i in range(len(iqcutNoBgd)):
            if iqcutNoBgd[i] < 0:
                iqcutNoBgd[i] = 0

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
        plt.scatter(effective_peak_x, effective_peak_y, label="Model",
                    marker='.', color='r')
        plt.legend(loc=0)

        residual += abs(iqcutNoBgd-sum_gauss(qcutM, *fit_data))
        sumRes = (sum(residual)/len(effective_peak_x))+zero_count*1000
        print("Sum of residual for " + models[fitIndx]['material_id'] + ": "
              + str(sumRes))

        plt.show()

    except RuntimeError:
        residual = 100000+zero_count*1000
        print("There weren't any fits")
