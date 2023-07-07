# -*- coding: utf-8 -*-
from random import choice
from random import randint
from string import ascii_uppercase

from pulse3D.constants import CARDIAC_STIFFNESS_FACTOR
from pulse3D.constants import CARDIAC_STIFFNESS_LABEL
from pulse3D.constants import MAX_CARDIAC_EXPERIMENT_ID
from pulse3D.constants import MAX_EXPERIMENT_ID
from pulse3D.constants import MAX_SKM_EXPERIMENT_ID
from pulse3D.constants import MAX_VARIABLE_EXPERIMENT_ID
from pulse3D.constants import MIN_EXPERIMENT_ID
from pulse3D.constants import ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR
from pulse3D.constants import SKM_STIFFNESS_FACTOR
from pulse3D.constants import SKM_STIFFNESS_LABEL
from pulse3D.constants import VARIABLE_STIFFNESS_LABEL
from pulse3D.utils import get_experiment_id
from pulse3D.utils import get_stiffness_factor
from pulse3D.utils import get_stiffness_label
import pytest


# checking for 384-well plate
def random_well_name():
    return choice(ascii_uppercase[0:13]) + str(randint(0, 23))


def random_well_name_in_row(row):
    return row + str(randint(0, 23))


def random_variable_stiffness_exp_id():
    return randint(MAX_SKM_EXPERIMENT_ID + 1, MAX_VARIABLE_EXPERIMENT_ID)


@pytest.mark.parametrize(
    "test_barcode_format", ["ML2022001{experiment_id:03d}", "ML22001{experiment_id:03d}-2"]
)
def test_get_experiment_id__return_correct_value(test_barcode_format):
    test_experiment_id = randint(MIN_EXPERIMENT_ID, MAX_EXPERIMENT_ID)
    test_barcode = test_barcode_format.format(experiment_id=test_experiment_id)
    assert get_experiment_id(test_barcode) == test_experiment_id


@pytest.mark.parametrize(
    "test_experiment_id,test_well_name,expected_stiffness_factor",
    [
        (MIN_EXPERIMENT_ID, random_well_name(), CARDIAC_STIFFNESS_FACTOR),
        (MAX_CARDIAC_EXPERIMENT_ID, random_well_name(), CARDIAC_STIFFNESS_FACTOR),
        (MAX_CARDIAC_EXPERIMENT_ID + 1, random_well_name(), SKM_STIFFNESS_FACTOR),
        (MAX_SKM_EXPERIMENT_ID, random_well_name(), SKM_STIFFNESS_FACTOR),
        (MAX_SKM_EXPERIMENT_ID + 1, "A1", ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["A"]),
        (
            MAX_VARIABLE_EXPERIMENT_ID,
            random_well_name_in_row("A"),
            ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["A"],
        ),
        (
            random_variable_stiffness_exp_id(),
            random_well_name_in_row("B"),
            ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["B"],
        ),
        (
            random_variable_stiffness_exp_id(),
            random_well_name_in_row("C"),
            ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["C"],
        ),
        (
            random_variable_stiffness_exp_id(),
            random_well_name_in_row("D"),
            ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR["D"],
        ),
        (MAX_VARIABLE_EXPERIMENT_ID + 1, random_well_name(), SKM_STIFFNESS_FACTOR),
        (MAX_EXPERIMENT_ID, random_well_name(), CARDIAC_STIFFNESS_FACTOR),
    ],
)
def test_get_stiffness_factor__returns_correct_value(
    test_experiment_id, test_well_name, expected_stiffness_factor
):
    assert get_stiffness_factor(test_experiment_id, test_well_name) == expected_stiffness_factor


@pytest.mark.parametrize(
    "test_experiment_id,expected_stiffness_label",
    [
        (MIN_EXPERIMENT_ID, CARDIAC_STIFFNESS_LABEL),
        (MAX_CARDIAC_EXPERIMENT_ID, CARDIAC_STIFFNESS_LABEL),
        (MAX_CARDIAC_EXPERIMENT_ID + 1, SKM_STIFFNESS_LABEL),
        (MAX_SKM_EXPERIMENT_ID, SKM_STIFFNESS_LABEL),
        (MAX_SKM_EXPERIMENT_ID + 1, VARIABLE_STIFFNESS_LABEL),
        (MAX_VARIABLE_EXPERIMENT_ID, VARIABLE_STIFFNESS_LABEL),
        (MAX_VARIABLE_EXPERIMENT_ID + 1, SKM_STIFFNESS_LABEL),
        (MAX_EXPERIMENT_ID, CARDIAC_STIFFNESS_LABEL),
    ],
)
def test_get_stiffness_label__returns_correct_value(test_experiment_id, expected_stiffness_label):
    assert get_stiffness_label(test_experiment_id) == expected_stiffness_label


@pytest.mark.parametrize("test_experiment_id", [-1, 1000])
def test_get_stiffness_factor__raises_value_error_if_invalid_experiment_id_given(test_experiment_id):
    with pytest.raises(
        ValueError, match=f"Experiment ID must be in the range 000-999, not {test_experiment_id}"
    ):
        get_stiffness_factor(test_experiment_id, random_well_name())
