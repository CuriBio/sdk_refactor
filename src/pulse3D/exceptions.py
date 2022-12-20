# -*- coding: utf-8 -*-
"""Generic exceptions for the Mantarray SW."""
from typing import Tuple


class UnrecognizedFilterUuidError(Exception):
    pass


class FilterCreationNotImplementedError(Exception):
    pass


class DataAlreadyLoadedInPipelineError(Exception):
    pass


class PeakDetectionError(Exception):
    pass


class TwoFeaturesInARowError(PeakDetectionError):
    """There must always be a peak in between valleys and vice-versa."""

    def __init__(
        self,
        back_to_back_points: Tuple[int, int],
        feature_name: str = "feature",
    ) -> None:
        prepend_msg = f"Two back-to-back {feature_name}s in a row were detected at timepoints: {back_to_back_points[0]} and {back_to_back_points[1]}\n"
        super().__init__(prepend_msg)


class TwoValleysInARowError(TwoFeaturesInARowError):
    """There must always be a peak in between valleys."""

    def __init__(
        self,
        back_to_back_points: Tuple[int, int],
    ) -> None:
        super().__init__(
            back_to_back_points,
            feature_name="valley",
        )


class TwoPeaksInARowError(TwoFeaturesInARowError):
    """There must always be a valley in between peaks."""

    def __init__(
        self,
        back_to_back_points: Tuple[int, int],
    ) -> None:
        super().__init__(
            back_to_back_points,
            feature_name="peak",
        )


class TooFewPeaksDetectedError(PeakDetectionError):
    pass


class NoRecordingFilesLoadedError(Exception):
    pass


class SubprotocolFormatIncompatibleWithInterpolationError(Exception):
    pass
