import json
from xrdphase import PhaseIdentification


piClass = PhaseIdentification()


def test_read_data():
    """
    Check that the first element of qcut is greater than startCut and the last
    element of qcut is less than 20
    """
    qcut, iqcut, numPeaks = piClass._split_read_data(fileName=
                                                     'xrdphase/tests/test_data/Ni.chi',
                                                     cutDataStart=2)
    assert qcut[0] >= 2, "Data is being cut wrong"
    assert qcut[-1] <= 20, "Data is being cut wrong"
    assert numPeaks == 11, "Peaks haven't been counted correctly"


def test_identify_phase():
    """Check that model_name is the correct model"""
    with open('xrdphase/tests/test_data/models.json') as json_file:
        models = json.load(json_file)
    qcut, iqcut, numPeaks = piClass._split_read_data(fileName=
                                                     'xrdphase/tests/test_data/Ni.chi',
                                                     cutDataStart=2)
    fitIndx, model_name = piClass._split_identify_phase(models, qcut, iqcut)
    assert model_name == 'mp-23', "Wrong model was picked"
