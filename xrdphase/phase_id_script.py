"""This class combines the functions in phase_id_functions into a class with
a function that determines the phase of the sample data. It also allows for
command line use.

"""

from phase_id_functions import get_structures, read_data
from phase_id_functions import identify_phase, show_correct_model
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

    Methods
    -------
    find_phase(apiKey, elementList, fileName, cutDataStart)
        Finds the correct phase/space group symbol of the sample data

    """

    def __init__(self, apiKey=None, elementList=None, fileName=None,
                 cutDataStart=None):
        if apiKey is not None:
            self.apiKey = apiKey
        if elementList is not None:
            self.elementList = elementList
        if fileName is not None:
            self.fileName = fileName
        if cutDataStart is not None:
            self.cutDataStart = cutDataStart

    def find_phase(self, apiKey, elementList, fileName, cutDataStart):
        models = get_structures(apiKey, elementList)
        qcut, iqcut = read_data(fileName, cutDataStart)
        fitIndx = identify_phase(models, qcut, iqcut)
        show_correct_model(models, fitIndx, qcut, iqcut)

    def _split_read_data(self, fileName, cutDataStart):
        qcut, iqcut = read_data(self.fileName, self.cutDataStart)
        return qcut, iqcut

    def _split_identify_phase(self, models, qcut, iqcut):
        fitIndx = identify_phase(self.models, self.qcut, self.iqcut)
        model_name = models[fitIndx]['material_id']
        return fitIndx, model_name


def main():
    fire.Fire(PhaseIdentification)


if __name__ == '__main__':
    main()
