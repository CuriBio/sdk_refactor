# -*- coding: utf-8 -*-
from random import randint

import numpy as np
from pulse3D.constants import CARDIAC_STIFFNESS_FACTOR
from pulse3D.constants import MILLI_TO_BASE_CONVERSION
from pulse3D.constants import NEWTONS_PER_MILLIMETER
from pulse3D.constants import SKM_STIFFNESS_FACTOR
from pulse3D.transforms import calculate_force_from_displacement
import pytest


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
