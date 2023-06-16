Changelog for Pulse3D
=====================

0.33.8 (2023-06-15)
-------------------

Added:
^^^^^^
- Support for stimulation subprotocol loops


0.33.7 (2023-06-05)
-------------------

Fixed:
^^^^^^
- Issue loading from DF when there are more stim timepoints than magnetometer timepoints
- Issue loading zipped xlsx files


0.33.6 (2023-05-18)
-------------------

Fixed:
^^^^^^
- Correct use of baselines in Twitch Velocity


0.33.5 (2023-05-17)
-------------------

Added:
^^^^^^
- Support for python 3.11


0.33.4 (2023-05-08)
-------------------

Changed:
^^^^^^^^
- Raise exception when mag finding is unable to converge due to bad data


0.33.3 (2023-05-08)
-------------------

Added:
^^^^^^
- Time window option to noise-based peak finding


0.33.2 (2023-05-04)
-------------------

Added:
^^^^^^
- Noise-based peak finding

0.33.1 (2023-05-04)
-------------------

Added:
^^^^^^
- Changes from 0.32.7


0.33.0 (2023-05-01)
-------------------

Fixed:
^^^^^^
- Correct use of baseline widths in Twitch AUC, Twitch Amplitude, and Twitch Velocities


0.32.7 (2023-05-02)
-------------------

**NOTE**: The date on this release is correct, it was released after 0.33.0, so 0.33.0 does not include these changes

Fixed:
^^^^^^
- Issue with ``to_dataframe`` outputting an empty DF when ``include_stim_data=True`` and no stim data present


0.32.6 (2023-04-28)
-------------------

Added:
^^^^^^
- Ability to ``PlateRecording`` to handle zip file containing multiple optical xlsx files


0.32.5 (2023-04-27)
-------------------

Fixed:
^^^^^^
- Correctly calculate upper x bound for snapshot graphs in ``create_waveform_charts``


0.32.4 (2023-04-21)
-------------------

Changed:
^^^^^^^^
- Re-add ability to override default magnet finding alg positions


0.32.3 (2023-04-04)
-------------------

Added:
^^^^^^
- Ability to ignore stim data in ``to_dataframe``


0.32.2 (2023-03-23)
-------------------

Fixed:
^^^^^^
- Skip reading metadata from missing wells when loading optical files


0.32.1 (2023-03-21)
-------------------

Fixed:
^^^^^^
- ``PlateRecording.to_dataframe`` when loaded from an xlsx input file


0.32.0 (2023-03-16)
-------------------

Added:
^^^^^^
- Accuracy improvements to the magnet finding algorithm

Changed:
^^^^^^^^
- Removed ability to override default magnet finding alg positions. This a temporary change will be readded in a future release


0.31.0 (2023-03-16)
-------------------

Fixed:
^^^^^^
- Update magnet finding algorithm submodule to latest


0.30.6 (2023-03-14)
-------------------

Fixed:
^^^^^^
- Removed noise filtering and other transforms on optical xlsx input files


0.30.5 (2023-03-13)
-------------------

Changed:
^^^^^^^^
- Stim subprotocols no longer continue being draw if they run longer than expected
- ``include_stim_protocols`` is now overrided to ``True`` if ``stim_waveform_format`` is given



0.30.4 (2023-03-01)
-------------------

Added:
^^^^^^
- Accuracy improvements to the magnet finding algorithm

Changed:
^^^^^^^^
- Filtering of data is now applied prior to the magnet finding algorithm instead of after


0.30.3 (2023-02-27)
-------------------

Added:
^^^^^^
- ``well_groups`` param added to ``PlateRecording`` to override the well groups in the H5 files
- Added ``Platemap Group Metrics`` to the  ``aggregate-metrics`` sheet after individual well metrics

Changed:
^^^^^^^^
- Removed ``PlateRecording.write_time_force_csv``


0.30.2 (2023-02-22)
-------------------

Changed:
^^^^^^^^
- ``peak_detector`` will now remove timepoints from outside the window provided by ``start_time`` and
  ``end_time`` before running the data through the peak finding algorithm. This is too ensure that problematic
  data outside the window does not interfere with peak finding.


0.30.1 (2023-02-07)
-------------------

Fixed:
^^^^^^
- Issue with double underscores in file names causing WellFiles to get initialized with the incorrect
  ``has_inverted_post_magnet`` value.


0.30.0 (2023-01-27)
-------------------

Added:
^^^^^^
- Initial support for PlateMaps, including a new section in the metadata sheet and the PlateMap Labels for
  each well on the aggregate metrics sheets


0.29.2 (2023-01-24)
-------------------

Fixed:
^^^^^^
- Calculation of twitch amplitude now correctly interpolates the baseline linearly between the
  Contraction 10% and Relaxation 90% points.


0.29.1 (2023-01-23)
-------------------

Fixed:
^^^^^^
- Calculation of twitch amplitude now uses Contraction 10% and Relaxation 90% points for baseline


0.29.0 (2022-12-22)
-------------------

Added:
^^^^^^
- Graphing of stimulator output waveforms:

  - "Overlayed" display option which graphs stim waveforms in the same chart as the tissue waveforms
  - "Stacked" display which graphs stim waveforms in a separate chart beneath the tissue waveforms chart

- Ability to interpolate stimulator output waveforms from start timepoints of each subprotocol


0.28.3 (2022-12-08)
-------------------
- Fix ``to_dataframe`` to include minimum timepoint

0.28.2 (2022-12-08)
-------------------
- Show aggregate metrics as long as 1 twitch is present

0.28.1 (2022-12-06)
-------------------
- Added option to add stim protocols sheet in analysis output

0.28.0 (2022-11-16)
-------------------
- Accuracy and performance improvements to the magnet finding algorithm
- ``PlateRecording``'s ``start_time`` and ``end_time`` parameters now take effect before running the magnet finding algorithm.
  These params are currently only intended to be used for recording snapshots. They have no effect on Beta 1 data
- Removed ``use_mean_of_baseline`` from ``PlateRecording`` since the alternative is never used

0.27.5 (2022-11-10)
-------------------
- Added Stim Lid Barcode to output file

0.27.4 (2022-11-08)
-------------------
- Add ability to flip waveform data of individual wells for Beta 1 data files

0.27.3 (2022-11-01)
-------------------
- Fix Post Stiffness factor being incorrectly reported in metadata sheet if an override value is provided

0.27.2 (2022-10-31)
-------------------
- Added Post Stiffness factor to metadata sheet of output
- Changed energy label from ``Energy (μJ)`` to ``Area Under Curve (μN * second)``

0.27.1 (2022-10-20)
-------------------
- Fixed issue with ``twitch_width_percents`` not being sorted

0.27.0 (2022-10-07)
-------------------
- Change magnet finding algorithm to account for 180 degree rotation of plates on V1 instrument
- Fix issue with trying to grab barcode from calibration files

0.26.1 (2022-10-05)
-------------------
- Added ability to pass kwargs to use in ``PlateRecording`` initialization through ``PlateRecording.from_directory``

0.26.0 (2022-10-04)
-------------------
- Added stiffness factor loading from barcode, and option to override the stiffness factor of the barcode
- Added ``Time From Peak to Relaxation 10 (seconds)`` to default output

0.25.4 (2022-09-20)
- Added new normalize_y_axis param to disable or enable y axis normalization

0.25.3 (2022-09-15)
-------------------
- Added support for multiple optical files in a zip folder

0.25.2 (2022-09-14)
-------------------
- Add static method ``from_dataframe`` to PlateRecording
- Add ``_load_dataframe`` method to PlateRecording
- Add ``get_windowed_peaks_valleys`` to peak_detection
- Changed ``continuous-waveforms`` excel sheet to begin at start of window of analysis if given, else 0

0.25.1 (2022-08-25)
-------------------
- Add twitch_widths to ``TwitchVelocity`` and ``TwitchAUC``

0.25.0 (2022-08-23)
-------------------
- Added the option to set custom y-axis for output graphs

0.24.9 (2022-08-23)
-------------------
- Added ``to_dataframe`` method to PlateRecording

0.24.8 (2022-08-15)
-------------------
- Added write_xlsx handling of single number input for width and prominence factors

0.24.7 (2022-08-10)
-------------------
- Added end_time and start_time params for PlateRecording class
- Added width and prominence factor to Pulse3D documentation

0.24.6 (2022-08-01)
-------------------
- Added width and prominence factors to API. Can now be called from the binder inside the write_excel function

0.24.5 (2022-07-14)
-------------------
- Added updated image for the twitch metrics diagram used in the documentation
- Added ability for user to add any twitch width instead of only multiples of 5
- Fixed delayed recording bug

0.24.4 (2022-07-12)
-------------------
- Added ``baseline_widths_to_use`` to ``write_xlsx`` args to replace existing baseline metric
- Default baseline metric changed to C10 and R90

0.24.1 (2022-06-21)
-------------------
- Rename constant for UUID value
- Add Apple M1 chip support


0.24.0 (2022-06-17)
-------------------
- Add support for V1 Mantarray data files


0.23.9 (2022-06-08)
-------------------
- Change ``write_xlsx`` to return name of generated output file


0.23.8 (2022-05-12)
-------------------
- Updated column and index values to well names and seconds for write_time_force_csv method

0.23.7 (2022-05-11)
-------------------
- Add write_time_force_csv method to PlateRecording
- Updated diagram png

0.23.6 (2022-04-14)
-------------------
- Fixed issue where desired twitch widths weren't being output in aggregate metrics sheet
- Fixed code snippets in documentation
- Changed output file name to include input file name
- Removed ``name`` param of ``write_xlsx`` function

0.23.5 (2022-04-07)
-------------------
- Added metadata for stim barcode

0.23.4 (2022-03-10)
-------------------
- Fix optical recording file loading
- Change indexing into excel spreadsheet rows

0.23.3 (2022-02-11)
-------------------
- Fix Beta 2 files analysis speed up

0.23.2 (2022-02-11)
-------------------
- Optimize metrics functions, lexsort issues, and dataframe pre-processing

0.23.1 (2022-02-11)
-------------------
- Fix Beta 2 files analysis speed up

0.23.0 (2022-02-10)
-------------------
- 10x speed up for analysis of Beta 2 files

0.22.4 (2022-02-09)
-------------------
- Add Beta 2 metadata UUIDs

0.22.3 (2022-02-09)
-------------------
- sort_index, not sort_value

0.22.2 (2022-02-09)
-------------------
- sort_index

0.22.1 (2022-02-09)
-------------------
- Convert time_points to pd.Series

0.22.0 (2022-02-07)
-------------------
- Incorporate windowed waveform-analysis

0.21.1 (2022-01-12)
-------------------
- Parameterized ``peak_detection.peak_detector`` for minimum prominence and width scaling
- Changed default scaling factors to make peak-finding more sensitive

0.20.2 (2022-01-12)
-------------------
- Fixed install issues

0.20.1 (2022-01-11)
-------------------
- Improved magnet finding algorithm performance

0.20.0 (2022-01-07)
-------------------
- Added Beta 2.2 support
- Fixed conversion of Beta 2.2 position data to force

0.19.0 (2021-12-08)
-------------------
- refactor, rename

0.18.1 (2021-10-20)
-------------------
- Fixed offset peak detection

0.17.1 (2021-09-24)
-------------------
- SkM metrics

0.16.1 (2021-07-21)
-------------------
- Multi zip

0.15.0 (2021-04-27)
-------------------
- Added Twitch Interval Irregularity metric to the per twitch metrics page and the aggregate metrics page


0.14.0 (2021-04-20)
-------------------
- Added Twitch Width metrics to the per twitch metrics sheet and aggregate metrics sheet
- Added Twitch Contraction adn Relaxation Coordinates to the per twitch metrics sheet
- Fixed twitch directionality to default to point upwards for force data


0.13.3 (2021-04-05)
-------------------
- Ignore hidden files when listing platereading files


0.13.2 (2021-03-29)
-------------------
- Bumped version to refresh MyBinder cache


0.13.1 (2021-03-23)
-------------------
- Bumped version to refresh MyBinder cache


0.13.0 (2021-03-19)
-------------------
- Added ability to analyze multiple recordings at once by traversing subdirectories


0.12.0 (2021-03-18)
-------------------
- Incorporated v0.7.0 of waveform-analysis, changing the units of metrics to force


0.11.0 (2021-03-03)
-------------------
- Added Twitch Relaxation Velocity and Contraction Velocity metrics to per twitch metrics sheet and aggregate metrics sheet


0.10.3 (2021-02-24)
-------------------
- Testing new publish workflow


0.10.2 (2021-02-17)
-------------------
- Incorporated v0.5.11 of waveform-analysis, patching some issues with peak detection


0.10.1 (2021-01-19)
-------------------
- Bumped Docker Container to 3.9.1-slim-buster
- Added message in Jupyter Notebook if not running the latest version


0.10.0 (2021-01-15)
-------------------
- Added twitch frequencies chart excel sheet.
- Added force frequency relationship chart excel sheet.


0.9.0 (2021-01-06)
------------------
- Added Area Under the Curve metric to per twitch metrics sheet and aggregate metrics sheet
- Fixed issue with interpolation values outside of the given boundaries for optical data.


0.8.2 (2020-12-29)
------------------

- Fixed issue with getting the incorrect well index from the well name for optical data.


0.8.1 (2020-12-20)
------------------

- Added Python 3.9 support.
- Added steps to documentation explaining how to analyze multiple zip files.
- Changed formatting of .xlsx output file names to match input the formatting
  of the input file names. A discrepancy still exists between the input and
  output file names, however.
- Added excel sheet for per twitch metrics.


0.8.0 (2020-11-11)
------------------

- Added excel sheet for full length charts.
- Fixed issue with pure noise files causing errors.


0.7.3 (2020-11-05)
------------------

- Fixed issue with twitches point up field for optical data.
- Fixed case sensitivity issue ('y' and 'Y' both work now).
- Fixed issue causing change of chart bounds to be tedious.
- Fixed Y axis label for optical data (now 'Post Displacement (microns)').
- Fixed many of the issues causing two consecutive relaxations to be
  detected incorrectly.
- Fixed interpolation bugs.
- Fixed documentation issues.
- Changed Sampling / Frame Rate from period in seconds to a rate in Hz.


0.7.1 (2020-10-20)
------------------

- Fixed issue with markers in optical data charts.


0.7.0 (2020-10-15)
------------------

- Added ability to analyze optical data entered in an excel template.
- Added firmware version to excel metadata sheet.


0.6.0 (2020-10-07)
------------------

- Added numbered steps to getting started documentation.
- Added ``contiuous-waveform-plots`` sheet to excel file generation.
  Currently, the only format for chart creation is a <= 10 second "snapshot" of
  the middle data points. It shows waveforms as well as Contraction and
  Relaxation markers on twitches.
- Added access to reference sensor data.
- Added performance improvements for accessing raw data.
- Added ability to upload zip files to Jupyter and updated ``Getting Started``
  documentation to show how to do so.
- Changed all interpolation to 100 Hz.
- Changed default filter for 1600 µs sampling period from Bessel Lowpass 30Hz
  to Butterworth Lowpass 30Hz.
- Fixed peak detection algorithm so it is less likely to report two
  contractions/relaxations of a twitch in a row.


0.5.0 (2020-09-21)
------------------

- Added logging to ``write_xlsx``.
- Added backwards compatibility with H5 file versions >= ``0.1.1``.


0.4.1 (2020-09-16)
------------------

- Added Jupyter getting started documentation.


0.4.0 (2020-09-16)
------------------

- Added support for MyBinder.
- Added Peak Detection Error handling.
- Added function to create stacked plot.


0.3.0 (2020-09-09)
------------------

- Added generation of Excel file with continuous waveform and aggregate metrics.
- Added SDK version number to metadata sheet in Excel file.
