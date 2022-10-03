# -*- coding: utf-8 -*-
from random import randint

import numpy as np
from pulse3D.constants import CARDIAC_STIFFNESS_FACTOR
from pulse3D.constants import MILLI_TO_BASE_CONVERSION
from pulse3D.constants import NEWTONS_PER_MILLIMETER
from pulse3D.constants import ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR
from pulse3D.constants import SKM_STIFFNESS_FACTOR
from pulse3D.transforms import calculate_force_from_displacement
from pulse3D.transforms import get_stiffness_factor
import pytest


def random_well_idx():
    return randint(0, 23)


def random_well_idx_in_row(row):
    row_idx = ord(row) - ord("A")
    return randint(0, 5) * 4 + row_idx


@pytest.mark.parametrize("test_stiffness_factor", [CARDIAC_STIFFNESS_FACTOR, SKM_STIFFNESS_FACTOR, None])
def test_calculate_force_from_displacement__returns_correct_values(test_stiffness_factor):
    test_displacement = np.array([randint(0, 1000000) for _ in range(10)], dtype=np.float64)
    test_timepoints = np.arange(len(test_displacement))

    kwargs = {}
    if test_stiffness_factor:
        kwargs["stiffness_factor"] = test_stiffness_factor

    force_arr = calculate_force_from_displacement(np.array([test_timepoints, test_displacement]), **kwargs)

    expected_stiffness_factor = test_stiffness_factor if test_stiffness_factor else CARDIAC_STIFFNESS_FACTOR
    expected_force = test_displacement * NEWTONS_PER_MILLIMETER * expected_stiffness_factor

    np.testing.assert_array_equal(force_arr[0, :], test_timepoints)
    np.testing.assert_array_almost_equal(force_arr[1, :], expected_force)


@pytest.mark.parametrize("in_mm", [True, False])
def test_calculate_force_from_displacement__returns_correct_values__with_in_mm_specified(in_mm):
    test_displacement = np.array([randint(0, 1000000) for _ in range(10)], dtype=np.float64)
    test_timepoints = np.arange(len(test_displacement))
    # arbitrarily choosing this stiffness factor
    test_stiffness_factor = CARDIAC_STIFFNESS_FACTOR

    force_arr = calculate_force_from_displacement(
        np.array([test_timepoints, test_displacement]), CARDIAC_STIFFNESS_FACTOR, in_mm=in_mm
    )

    expected_force = test_displacement * NEWTONS_PER_MILLIMETER * test_stiffness_factor
    if not in_mm:
        expected_force *= MILLI_TO_BASE_CONVERSION

    np.testing.assert_array_equal(force_arr[0, :], test_timepoints)
    np.testing.assert_array_almost_equal(force_arr[1, :], expected_force)


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
