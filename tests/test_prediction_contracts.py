import numpy as np
import pytest
from repro_agent.harness import independence, nullius_io


@pytest.mark.parametrize('values', [[0.2,1.5],[-1,0],[float('nan'),1],[float('inf'),1]])
def test_node_output_never_silently_truncates_probabilities_or_invalid_values(values):
    with pytest.raises(ValueError): nullius_io._node_values(np.array(values),'pred',2)


def test_valid_integer_classes_and_finite_logits_are_preserved():
    assert nullius_io._node_values(np.array([0.,1.]),'pred',2).tolist()==[0,1]
    assert nullius_io._node_values(np.array([[0.2,0.8],[0.9,0.1]]),'pred',2).tolist()==[1,0]


@pytest.mark.parametrize('rows', ['-1_0_0,0.4','0_-1_0,0.4','0_0_-1,0.4','0_0.5_0,0.4','0_96_0,0.4',
                                 '0_0_7,0.4','0_0_0,nan','0_0_0,0.4\n0_0_0,0.5'])
def test_independence_probe_rejects_corrupt_ids_instead_of_overwriting_or_truncating(rows):
    with pytest.raises(ValueError): independence.parse_predictions(('id,prediction\n'+rows+'\n').encode(),1)
