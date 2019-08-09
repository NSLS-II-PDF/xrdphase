"""This is ACE - The Automated Crystallography Expert. This class combines the
functions in phase_id_functions into a class with a function that determines
the phase of the sample data. It also allows for command line use.

"""

from xrdphase.phase_id_functions import get_structures, read_data
from xrdphase.phase_id_functions import identify_phase, show_correct_model
from xrdphase.phase_id_functions import get_NN, identify_phase_nn
from xrdphase.training_NN_funcs import trainNN, pickle_nn
import fire


class PhaseIdentification:

    """Class constructor

    Initializes class attributes. All four attributes are required to use
    find_phase

    Parameters
    ----------
    `apiKey` : str
               This is your API key for the Materials Project database.
    `elementList` : str or list of str
                    If using in Jupyter notebook, leave this as a list. If
                    using in the command line, put the list of elements in
                    single quotes and the elements in double quotes or vice
                    versa.
    `fileName` : str
                 This is your sample data file. You need to include the
                 path to it
    `cutDataStart` : float
                     This determines where the beginning of the sample data
                     will be after being cut
    `q_dep_shift` : float, optional
                    This multiplies the shift amount to every Q value.
                    Default=1.0
    `const_shift` : float, optional
                    This adds the shift amount to every model Q value
                    Default=0.0

    Methods
    -------
    find_phase(apiKey, elementList, fileName, cutDataStart)
        Finds the correct phase/space group symbol of the sample data
    find_phase_nn(apiKey, elementList, fileName, cutDataStart)
        Finds the correct phase/space group symbol of the sample data using a
        neural network. Performs faster than `find_phase`.

    """

    def __init__(self, apiKey=None, elementList=None, fileName=None,
                 cutDataStart=None, q_dep_shift=None, const_shift=None):
        if apiKey is not None:
            self.apiKey = apiKey
        if elementList is not None:
            self.elementList = elementList
        if fileName is not None:
            self.fileName = fileName
        if cutDataStart is not None:
            self.cutDataStart = cutDataStart
        if q_dep_shift is not None:
            self.q_dep_shift = q_dep_shift
        if const_shift is not None:
            self.const_shift = const_shift

    def find_phase(self, apiKey, elementList, fileName, cutDataStart,
                   q_dep_shift=1.0, const_shift=0.0):
        models = get_structures(apiKey, elementList)
        qcut, iqcut, _ = read_data(fileName, cutDataStart)
        fitIndx = identify_phase(models, qcut, iqcut, q_dep_shift,
                                 const_shift)
        if fitIndx is None:
            pass
        else:
            show_correct_model(models, fitIndx, qcut, iqcut, const_shift,
                               q_dep_shift)

    def find_phase_nn(self, apiKey, elementList, fileName, cutDataStart,
                      q_dep_shift=1.0, const_shift=0.0):
        models = get_structures(apiKey, elementList)
        qcut, iqcut, numPeaks = read_data(fileName, cutDataStart)
        clf = get_NN()
        bestMatch = identify_phase_nn(models, qcut, iqcut, clf, numPeaks,
                                      q_dep_shift, const_shift)
        if bestMatch is None:
            pass
        else:
            show_correct_model(models, bestMatch, qcut, iqcut, const_shift,
                               q_dep_shift)

    def _create_NN(self):
        # This should only be run if the training set needs to be modified
        clf = trainNN()
        pickle_nn(clf)

    def _split_read_data(self, fileName, cutDataStart):
        qcut, iqcut, numPeaks = read_data(self.fileName, self.cutDataStart)
        return qcut, iqcut, numPeaks

    def _split_identify_phase(self, models, qcut, iqcut):
        fitIndx = identify_phase(self.models, self.qcut, self.iqcut)
        model_name = models[fitIndx]['material_id']
        return fitIndx, model_name


def main():
    fire.Fire(PhaseIdentification)


if __name__ == '__main__':
    main()
