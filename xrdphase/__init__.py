
# from ._version import get_versions
# t`__version__ = get_versions()['version']
# del get_versions

from .phase_id_functions import get_structures, read_data
from .phase_id_functions import identify_phase, show_correct_model


class PhaseIdentification:
    def find_phase(self, apiKey, elementList, fileName, cutDataStart):
        models = get_structures(apiKey, elementList)
        qcut, iqcut = read_data(fileName, cutDataStart)
        fitIndx = identify_phase(models, qcut, iqcut)
        show_correct_model(models, fitIndx, qcut, iqcut)

    def _split_read_data(self, fileName, cutDataStart):
        qcut, iqcut = read_data(fileName, cutDataStart)
        return qcut, iqcut

    def _split_identify_phase(self, models, qcut, iqcut):
        fitIndx = identify_phase(models, qcut, iqcut)
        model_name = models[fitIndx]['material_id']
        return fitIndx, model_name
