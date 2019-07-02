# going to test using python fire
from phase_identification_functions import get_structures, read_data
from phase_identification_functions import identify_phase, show_correct_model
import fire

class PhaseIdentification:
    def find_phase(self, apiKey, elementList, fileName, cutDataStart):
        models = get_structures(apiKey, elementList)
        qcut, iqcut = read_data(fileName, cutDataStart)
        fitIndx, modelList = identify_phase(models, qcut, iqcut)
        show_correct_model(models, fitIndx, modelList, qcut, iqcut)


        # self._find_phase_no_api(models, fileName, cutDataStart)


    def _split_read_data(self, fileName, cutDataStart):
        qcut, iqcut = read_data(fileName, cutDataStart)
        return qcut, iqcut


    def _split_identify_phase(self, models, qcut, iqcut):
        fitIndx, modelList = identify_phase(models, qcut, iqcut)
        model_name = models[modelList[fitIndx]]['material_id']
        return fitIndx, modelList, model_name


def main():
    fire.Fire(PhaseIdentification)

if __name__ == '__main__':
    main()
