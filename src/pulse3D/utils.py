# -*- coding: utf-8 -*-
"""General utility/helpers."""
import logging
import math
from typing import Any
from typing import Tuple
from typing import Union

from nptyping import NDArray

from .constants import CARDIAC_STIFFNESS_FACTOR
from .constants import MAX_CARDIAC_EXPERIMENT_ID
from .constants import MAX_EXPERIMENT_ID
from .constants import MAX_SKM_EXPERIMENT_ID
from .constants import MAX_VARIABLE_EXPERIMENT_ID
from .constants import MIN_EXPERIMENT_ID
from .constants import ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR
from .constants import SKM_STIFFNESS_FACTOR
from .constants import TWENTY_FOUR_WELL_PLATE


logger = logging.getLogger(__name__)


def get_experiment_id(barcode: str) -> int:
    if "-" in barcode:
        barcode = barcode.split("-")[0]
    return int(barcode[-3:])


def get_stiffness_factor(barcode_experiment_id: int, well_idx: int) -> int:
    if not (MIN_EXPERIMENT_ID <= barcode_experiment_id <= MAX_EXPERIMENT_ID):
        raise ValueError(f"Experiment ID must be in the range 000-999, not {barcode_experiment_id}")

    if barcode_experiment_id <= MAX_CARDIAC_EXPERIMENT_ID:
        return CARDIAC_STIFFNESS_FACTOR
    if barcode_experiment_id <= MAX_SKM_EXPERIMENT_ID:
        return SKM_STIFFNESS_FACTOR
    if barcode_experiment_id <= MAX_VARIABLE_EXPERIMENT_ID:
        well_row_label = TWENTY_FOUR_WELL_PLATE.get_well_name_from_well_index(well_idx)[0]
        return ROW_LABEL_TO_VARIABLE_STIFFNESS_FACTOR[well_row_label]
    # if experiment ID does not have a stiffness factor defined (currently 300-999) then just use the value for Cardiac
    return CARDIAC_STIFFNESS_FACTOR


def truncate_float(value: float, digits: int) -> float:
    if digits < 1:
        raise ValueError("If truncating all decimals off of a float, just use builtin int() instead")
    # from https://stackoverflow.com/questions/8595973/truncate-to-three-decimals-in-python
    stepper = 10.0**digits
    return math.trunc(stepper * value) / stepper


def truncate(
    source_series: NDArray[(1, Any), float], lower_bound: Union[int, float], upper_bound: Union[int, float]
) -> Tuple[int, int]:
    """Match bounding indices of source time-series with reference time-series.

    Args:
        source_series (NDArray): time-series to truncate
        lower_bound/upper_bound (float): bounding times of a reference time-series

    Returns:
        first_idx (int): index corresponding to lower bound of source time-series
        last_idx (int): index corresponding to upper bound of source time-series
    """
    first_idx, last_idx = 0, len(source_series) - 1

    # right-truncation
    while source_series[last_idx] > upper_bound:
        last_idx -= 1

    # left-truncation
    while source_series[first_idx] < lower_bound:
        first_idx += 1

    return first_idx, last_idx


def xl_col_to_name(col, col_abs=False):
    """Convert a zero indexed column cell reference to a string.

    Args:
       col:     The cell column. Int.
       col_abs: Optional flag to make the column absolute. Bool.

    Returns:
        Column style string.
    """
    col_num = col
    if col_num < 0:
        raise ValueError("col arg must >= 0")

    col_num += 1  # Change to 1-index.
    col_str = ""
    col_abs = "$" if col_abs else ""

    while col_num:
        # Set remainder from 1 .. 26
        remainder = col_num % 26
        if remainder == 0:
            remainder = 26
        # Convert the remainder to a character.
        col_letter = chr(ord("A") + remainder - 1)
        # Accumulate the column letters, right to left.
        col_str = col_letter + col_str
        # Get the next order of magnitude.
        col_num = int((col_num - 1) / 26)

    return col_abs + col_str
