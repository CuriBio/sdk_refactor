# -*- coding: utf-8 -*-
from random import randint

from pulse3D.constants import MAX_EXPERIMENT_ID
from pulse3D.constants import MIN_EXPERIMENT_ID
from pulse3D.utils import get_experiment_id
import pytest


@pytest.mark.parametrize("test_barcode_format", ["ML2022001{experiment_id}", "ML22001{experiment_id}-2"])
def test_get_experiment_id__return_correct_value(test_barcode_format):
    test_experiment_id = randint(MIN_EXPERIMENT_ID, MAX_EXPERIMENT_ID)
    test_barcode = test_barcode_format.format(experiment_id=test_experiment_id)
    assert get_experiment_id(test_barcode) == test_experiment_id
