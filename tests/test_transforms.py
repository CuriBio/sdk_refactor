# -*- coding: utf-8 -*-
from random import randint

from pulse3D.constants import CARDIAC_STIFFNESS_FACTOR
from pulse3D.constants import ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR
from pulse3D.constants import SKM_STIFFNESS_FACTOR
from pulse3D.transforms import get_stiffness_factor
import pytest


def random_well_idx():
    return randint(0, 23)


def random_well_idx_in_row(row):
    row_idx = ord(row) - ord("A")
    return randint(0, 5) * 4 + row_idx


@pytest.mark.parametrize(
    "test_experiment_id,test_well_idx,expected_stiffness_factor",
    [
        (0, random_well_idx(), CARDIAC_STIFFNESS_FACTOR),
        (99, random_well_idx(), CARDIAC_STIFFNESS_FACTOR),
        (100, random_well_idx(), SKM_STIFFNESS_FACTOR),
        (199, random_well_idx(), SKM_STIFFNESS_FACTOR),
        (200, random_well_idx_in_row("A"), ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["A"]),
        (299, random_well_idx_in_row("A"), ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["A"]),
        (randint(200, 299), random_well_idx_in_row("B"), ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["B"]),
        (randint(200, 299), random_well_idx_in_row("C"), ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["C"]),
        (randint(200, 299), random_well_idx_in_row("D"), ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["D"]),
        (300, random_well_idx(), CARDIAC_STIFFNESS_FACTOR),
        (999, random_well_idx(), CARDIAC_STIFFNESS_FACTOR),
    ],
)
def test_get_stiffness_factor__returns_correct_value(
    test_experiment_id, test_well_idx, expected_stiffness_factor
):
    assert get_stiffness_factor(test_experiment_id, test_well_idx) == expected_stiffness_factor


@pytest.mark.parametrize("test_experiment_id", [-1, 1000])
def test_get_stiffness_factor__raises_value_error_if_invalid_experiment_id_given(test_experiment_id):
    with pytest.raises(
        ValueError, match=f"Experiment ID must be in the range 000-999, not {test_experiment_id}"
    ):
        get_stiffness_factor(test_experiment_id, random_well_idx())
