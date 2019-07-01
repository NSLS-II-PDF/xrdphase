import numpy as np
import matplotlib.pyplot as plt
from pymatgen import MPRester
from free_pdf import find_nearest, cut_data, read_index_data_smart
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def get_structures(apiKey, elements):
    """Gets structures from Materials Project database.
    Pass in api key, a list of elements in the sample,
    and the number of different elements in the sample"""
    mpr = MPRester(apiKey)
    numElements = len(elements)
    models = mpr.query(criteria={"elements": {"$all": [*elements]},
                                 "nelements": numElements},
                       properties=['xrd.Cu', 'material_id', 'spacegroup'])
    return models


def read_data(fileName, startCut):
    """Reads in the data from the sample chi file and makes an
    initial cut at the beginning of the data
    Used to cut out air scatter and the beam stop from the start of the
    data"""
    q, iq = read_index_data_smart(fileName)
    qcut, iqcut = cut_data(q, iq, startCut, 20)
    return qcut, iqcut


def bgd_func(qcut, iqcutStart, iqcutEnd):
    "Background function - Linear"
    "Pass in qcut values and the start and end points of iqcut"
    a1 = (iqcutEnd-iqcutStart)/(qcut[-1]-qcut[0])
    a0 = iqcutStart-(qcut[0]*(a1))
    return a0 + a1*qcut


def sum_gauss(x, *params):
    "Gaussian function to fit the data peaks"
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        cen = params[i]
        wid = params[i+1]
        amp = params[i+2]
        y = y + np.exp(-((x-cen)**2)/(2.*wid**2))*amp
    return y


def convert_tth_to_q(tth, lam=1.5406):
    "Converts 2theta values in Materials Project models to q"
    q = (np.pi*4.0/lam)*np.sin(tth/360*np.pi)
    return q


def plot_background_and_structure(models, modelIndx, qcut, iqcut,
                                  plot_this=False):
    """Cuts the sample data to match where a single model's peaks are,
    then finds and removes the background. If plot_this is set to True,
    plots the cut data and the calculated background, then the sample data
    with no background and the position of the model peaks"""
    model_num = models[modelIndx]['xrd.Cu']['pattern']
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

    if plot_this:
        plt.figure()
        plt.plot(qcutM, iqcutM, label="Data")
        plt.plot(qcutM, bgd, label="Background")
        plt.figure()
        plt.plot(qcutM, iqcutNoBgd, label="Data - No Background")
        plt.scatter(effective_peak_x, effective_peak_y, label="Model",
                    marker='.', color='r')
        plt.legend(loc=0)


def identify_phase(models, qcut, iqcut):
    """Identifies the most likely spacegroup symbol and material_id based
    on the sample data and Materials Project models.
    Pass in the list of models, qcut, and iqcut.
    This function attempts to fit a curve to the data based on the position
    and amplitudes of the models.
    It ignores models that look like they have too many peaks and it
    looks for the model with the
    smallest residual between the data and the fitted curve"""
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

        # cuts out models with way too many peaks
        if len(effective_peak_x) <= 2*len(qcutM[peaks]):
            plt.figure()
            plt.plot(qcutM, iqcutNoBgd)
            plt.scatter(effective_peak_x, effective_peak_y, label="Model",
                        marker='.', color='r')
            plt.plot(qcutM, sum_gauss(qcutM, *initial_guess),
                     label="Initial Guess")
            plt.title(models[j]['material_id'] + "\t" +
                      models[j]['spacegroup']['symbol'])

            print("zero count is " + str(zero_count))

            try:
                fit_data, _ = curve_fit(sum_gauss, qcutM, iqcutNoBgd,
                                        p0=initial_guess,
                                        bounds=([low_lims, high_lims]),
                                        maxfev=100)
                plt.plot(qcutM, iqcutNoBgd, label="Data")
                plt.plot(qcutM, sum_gauss(qcutM, *fit_data), label="Fit")
                plt.legend(loc=0)

                print('done fitting the curve')

                residual += abs(iqcutNoBgd-sum_gauss(qcutM, *fit_data))
                sumRes = sum(residual)+zero_count*1000
                print("Sum of residual: " + str(sumRes))
                resSumList.append(sumRes)

                modelList.append(j)

            except RuntimeError:
                residual = 100000+zero_count*1000
                print("Curve fit probably failed here")
                print("Bad curve residual: " + str(residual))
        else:
            print("I think curve fit will fail for " +
                  models[j]['material_id'] +
                  ". There might be too many peaks")

    # find index of smallest residual
    smallIndx = resSumList.index(min(resSumList))
    print("I think you're looking at " +
          models[modelList[smallIndx]]['spacegroup']['symbol'] +
          ', ' + models[modelList[smallIndx]]['material_id'])
