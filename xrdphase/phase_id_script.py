# uses python fire for command line interface
from phase_identification_functions import *
import fire

class PhaseIdentification:
    def find_phase(self, apiKey, elementList, fileName, cutDataStart):
        models = get_structures(apiKey, elementList)
        qcut, iqcut = read_data(fileName, cutDataStart)
        fitIndx, modelList = identify_phase(models, qcut, iqcut)
        show_correct_model(models, fitIndx, modelList, qcut, iqcut)

def main():
    fire.Fire(PhaseIdentification)

if __name__ == '__main__':
    main()
