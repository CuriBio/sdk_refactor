.. _using-locally:

Using the SDK Locally
=====================

The CuriBio SDK can be used locally to analyze multiple recordings in a single batch.

Downloading Python and Updated SDK Version
------------------------------------------

Ensure that you have Python downloaded and are updated to the newest version of the SDK.
The SDK is compatible with Python versions 3.7, 3.8, and 3.9 but 3.9 is recommended.
Click |python_link| to navigate to further instructions for downloading Python.

.. |python_link| raw:: html

   <a href="https://www.python.org/downloads/" target="_blank">here</a>

The update to the newest version of the SDK can be downloaded by running::

    pip install curibio.sdk --upgrade

Running Batch Analysis
----------------------

1. Open a command prompt. If you are unfamiliar with a command prompt, click |command_prompt_link|
for more information.

.. |command_prompt_link| raw:: html

   <a href="https://www.lifewire.com/command-prompt-2625840" target="_blank">here</a>

2. Then, in the command prompt, navigate to the folder that contains all the subfolders
of your recordings (using the 'cd' change directory command). If you are unfamiliar with
the change directory command, click |cd_link| for more information.

.. |cd_link| raw:: html

   <a href="https://www.digitalcitizen.life/command-prompt-how-use-basic-commands/" target="_blank">here</a>

3. Run this command (the part in quotes is a simple python script)::

    python3 -c "from curibio.sdk import check_if_latest_version, create_xlsx_for_all_recordings; check_if_latest_version(); create_xlsx_for_all_recordings()"

As it runs, you should see updates of something like "Analyzing recording 3 of 25" to let you
know how it's progressing.

This should work so long as the H5 files in any individual folder are all from the same recording
(this should happen by default if you're copy/pasting folders from the main Mantarray recordings folder
into somewhere else you're organizing your experiments).  You can have as many "organizational" folders
as you want with no H5 files, and then just put the folder containing the H5 files in at some point.

You can navigate to the top-level folder and then run the script. Or, if you just wanted to run it
for a particular subfolder (and all its nested subfolders), you can navigate there and run it too.

All the Excel files will be generated at the root folder where you ran the script from
and are labeled with the plate barcode and timestamp (so that you don't have to go hunting for them
in the subfolders).

Troubleshooting
---------------

If the command in Step 3 of Running Batch Analysis isn't working, try using python
instead of python3::

    python -c "from curibio.sdk import check_if_latest_version, create_xlsx_for_all_recordings; check_if_latest_version(); create_xlsx_for_all_recordings()"
