import pickle
import random
import numpy as np
from sklearn.neural_network import MLPClassifier

def convert_tth_to_q(tth, lam=1.5406):
    q = (np.pi*4.0/lam)*np.sin(tth/360*np.pi)
    return q

#gauss function
def gauss_func(x,wid,cen,amp):
    return np.exp(-((x-cen)**2.)/(2.*wid**2)) * amp

def trainNN():
    # want to train this once and pickle it to use it later
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
    filename = './nnMLPClass'
    outfile = open(filename, 'wb')
    pickle.dump(clf, outfile)
    outfile.close()
