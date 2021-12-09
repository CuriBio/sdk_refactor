# -*- coding: utf-8 -*-
# """Fixtures for testing the metrics."""
# import os
# import pickle
# import pytest

# from peak_detection import peak_detector
# from constants import MICRO_TO_BASE_CONVERSION

# from stdlib_utils import get_current_file_abs_directory

# from .fixtures_utils import fixture_raw_generic_well_a1
# from .fixtures_utils import fixture_raw_generic_well_a2
# from .fixtures_utils import fixture_sample_tissue_reading
# from .fixtures_utils import fixture_sample_reference_reading

# __fixtures__ = [
#     fixture_raw_generic_well_a1,
#     fixture_raw_generic_well_a2,
#     fixture_sample_reference_reading,
#     fixture_sample_tissue_reading
# ]

# PATH_OF_CURRENT_FILE = get_current_file_abs_directory()
# PATH_TO_DATASETS = os.path.join(PATH_OF_CURRENT_FILE, "datasets")


# @pytest.fixture(scope="session", name="generate_twitch_amplitude")
# def fixture_generate_twitch_amplitude():
#     """For regression tests on twitch amplitude."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_amplitude.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_fraction_amplitude")
# def fixture_generate_twitch_fraction_amplitude():
#     """For regression tests on fraction of max twitch amplitude."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_fraction_amplitude.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_width")
# def fixture_generate_twitch_width():
#     """For regression tests on twitch widths."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_width.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_auc")
# def fixture_generate_twitch_auc():
#     """For regression tests on twitch AUC."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_auc.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_velocity_relaxation")
# def fixture_generate_twitch_velocity_relaxation():
#     """For regression tests on twitch velocity (relaxation)."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_velocity_relaxation.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_velocity_contraction")
# def fixture_generate_twitch_velocity_contraction():
#     """For regression tests on twitch velocity (contraction)."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_velocity_contraction.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_period")
# def fixture_generate_twitch_period():
#     """For regression tests on twitch period."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_period.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_frequency")
# def fixture_generate_twitch_frequency():
#     """For regression tests on twitch frequency."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_frequency.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_irregularity")
# def fixture_generate_twitch_irregularity():
#     """For regression tests on twitch irregularity."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_irregularity.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_peak_time_relaxation")
# def fixture_generate_twitch_peak_time_relaxation():
#     """For regression tests on twitch time-to-peak (relaxation)."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_peak_time_relaxation.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_peak_to_baseline")
# def fixture_generate_twitch_peak_to_baseline():
#     """For regression tests on twitch full relaxation time."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_peak_to_baseline.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_peak_time_contraction")
# def fixture_generate_twitch_peak_time_contraction():
#     """For regression tests on twitch time-to-peak (contraction)."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_peak_time_contraction.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="session", name="generate_twitch_baseline_to_peak")
# def fixture_generate_twitch_baseline_to_peak():
#     """For regression tests on twitch full contraction time."""
#     with open(os.path.join(PATH_TO_DATASETS, "metrics", "twitch_baseline_to_peak.pkl"), "rb") as f:
#         return pickle.load(f, fix_imports=True)


# @pytest.fixture(scope="function", name="generic_well_features")
# def fixture_generic_well_features(raw_generic_well_a1, raw_generic_well_a2):
#     """Load peak/valley indices, filtered data, and twitch indices."""
#     no_filter_pipeline_template = PipelineTemplate(tissue_sampling_period=1)
#     pipeline = no_filter_pipeline_template.create_pipeline()
#     pipeline.load_raw_magnetic_data(raw_generic_well_a1, raw_generic_well_a2)

#     filtered_data = pipeline.get_noise_filtered_magnetic_data()
#     peak_and_valley_indices = pipeline.get_peak_detection_results()
#     twitch_indices = peak_detection.find_twitch_indices(peak_and_valley_indices)

#     return [filtered_data, peak_and_valley_indices, twitch_indices]
