Changelog for Curi Bio Software Development Kit
===============================================
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
