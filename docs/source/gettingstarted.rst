.. _gettingstarted:

Jupyter Notebooks
=================

Jupyter is the environment that CuriBio SDK is designed to be used in. It allows creation
of Python Notebooks which consist of Code Cells (also referred to as just 'cells') that contain Python code,
each of which can be executed independently of others.

.. image:: images/twitch_metrics_diagram.png
    :width: 600

Getting Started with Jupyter
----------------------------

Click |mybinder_link| to navigate to the online
notebook.

.. |mybinder_link| raw:: html

   <a href="https://jupyter-sdk.curibio.com" target="_blank">here</a>

You should land on a page that looks like this:

.. image:: images/mybinder_loading_page.png
    :width: 600

It may take a few minutes to load the notebook. Once it's loaded you should see this page:

.. image:: images/fresh_nb.png
    :width: 600

Each block of code is a code cell. When a code cell is running, you will see this to
the left of it:

.. image:: images/running_cell.png
    :width: 100

When a cell completes execution, the star will become a number:

.. image:: images/finished_cell.png
    :width: 100

This number corresponds to the order the cells are run in. For this example,
there are only 3 cells, so if they are all ran in order the last cell should
have a 3 next to it. If a cell is re-run, the number will change.


Working With the SDK
====================

This section will demonstrate how to upload H5 files to Jupyter, convert them to
an excel sheet, and then download the converted files.


Uploading H5 Files
------------------

1. To begin uploading H5 files, click the Jupyter logo in the top left corner to
   navigate to the file explorer page:

.. image:: images/jupyter.png
    :width: 600

You should now be on this page listing all the folders and files currently in the environment:

.. image:: images/fresh_files_page.png
    :width: 600

2. Click on ``my-data``. You should now be in the ``my-data`` folder:

.. image:: images/my_data.png
    :width: 600

3. Click on the upload button near the top right and select the files you wish to upload.
   You may see an upload button next to each file you selected to upload.
   If this happens, just click the new upload button next to each file to complete the process.

.. image:: images/file_upload_button.png
    :width: 600

When the uploads complete, the page should look like this:

.. image:: images/uploaded_files.png
    :width: 600

Alternatively, you can upload multiple files as a ``zip`` file to speed up the
upload process. To do so on Windows, just select the local files you wish to zip.
Then, right click on the selection, and choose ``Send to`` ->
``Compressed (zipped) folder``:

.. image:: images/zip_menu.png
    :width: 600

After zipping you should see the ``zip`` file. This file will likely have the same
name as one the the files you zipped as shown here:

.. image:: images/zipped_files.png
    :width: 600

You can rename this ``zip`` file if you'd like to before uploading. Once the
files are zipped together, remember to only upload the ``zip`` file.

4. Once you finish uploading the file(s), click the tabs icon in the far left vertical menu here:

.. image:: images/open_notebook.png
    :width: 600

5. To return to the notebook click on intro.pynb here:

.. image:: images/load_notebook.png
    :width: 600


Note about uploading zip files
------------------------------

Only one zip file can be in the ``my-data`` folder at a time.
If you need to analyze multiple zip files:

1. Follow the steps in the next section to export the data from the first zip file.

2. Navigate back to the ``my-data`` folder and remove all the items in it.

3. Upload the next zip file.

This process can be repeated for any number of zip files. Alternatively, you can open
a new notebook for each other zip file you want to analyze.


Exporting Data to an Excel File and Downloading
-----------------------------------------------

1. Starting from the main files page, navigate back to the notebook
   page by clicking on ``intro.ipynb``.

2. Before running any code cells, you will need to update the file location.
   Change the line::

      recording = PlateRecording.from_directory('./sample-data')

   to::

      recording = PlateRecording.from_directory('./my-data')

3. You can now begin running the code. To do so, click ``Cell`` near the top left, then click ``Run All``:

.. image:: images/cell_run_all.png
    :width: 600

If there are many files, it may take a minute or two to convert all of them.
Progress messages will be printed to indicate the code is working and not frozen.
When all cells complete execution there should be a number next to every cell.
You will also see a message printed underneath the last cell indicating that
writing to the ``.xlsx`` file is complete:

.. image:: images/finished_cells.png
    :width: 600

4. Click on the Jupyter Logo in the top left of the page again to
go back to the files page. You should should now see a new ``.xlsx`` file. The
name of the file should contain the barcode of the plate the data was recorded from.

5. To download, check the box to the left of the file and then press ``download``
near the top left.

.. image:: images/download_screen.png
    :width: 600

When navigating away from the page you may see the following pop-up dialog:

.. image:: images/leave_page.png
    :width: 600

If all analysis is done and all files have been downloaded,
then it is OK to press ``leave``.

Adding Arguments to write_xlsx()
--------------------------------

If you would like to add twitch width values to the per twitch metrics sheet and aggregate metrics
sheet you can do so by adding a ``twitch_width_values`` argument to ``write_xlsx()``. There are five possible twitch widths that an be included which
are 10, 25, 50, 75, 90. You can include any combination of the five or none at all. However if a value is included
other than the five listed, an error will occur. If none are specified, then by default all will
be shown. An example of adding this argument is::

    write_xlsx(r, twitch_width_values=(10, 25, 90))

or::

    write_xlsx(r, twitch_width_values=(10,))

If you would like to include the twitch coordinate values for the twitch widths, that can be done so
by including a ``show_twitch_coordinate_values`` argument in ``write_xlsx()``. In order to do so the
argument must be set to ``True`` as by default it is set to ``False``. If it is set to ``True``
then the twitch coordinate values that will be included will be for the twitch widths that are specified or if it isn't
specified then all will be included. This metric will be included in the per twitch metrics sheet. An example of
this is::

    write_xlsx(r, twitch_width_values=(10, 25, 90), show_twitch_coordinate_values=True)

or::

    write_xlsx(r, show_twitch_coordinate_values=True)

If you would like to include the twitch time difference metric (the time difference between various
points on the curve and the peak) on the per twitch metrics sheet then you can do so by including a ``show_twitch_time_diff_values``
argument in ``write_xlsx()``. In order to do so the argument must be set to ``True`` as by default it is set to ``False``.
If it is set to ``True`` then the twitch time difference values that will be included will be for the twitch widths that are specified or if it isn't
specified then all will be included. An example of this is::

    write_xlsx(r, twitch_width_values=(10, 25, 90), show_twitch_coordinate_values=True, show_twitch_time_diff_values=True)

or::

    write_xlsx(r, show_twitch_time_diff_values=True)
