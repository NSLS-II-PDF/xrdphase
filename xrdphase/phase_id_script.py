from phase_identification_functions import get_structures, read_data
from phase_identification_functions import identify_phase, show_correct_model
import fire


class PhaseIdentification:

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
