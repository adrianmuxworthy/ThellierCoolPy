# ThelllierCoolPy
A cooling-rate correction tool for Thellier-type paleointensity data. Written in Python.

Instructions:

Python packages required to run this program:

numpy (pip install numpy) numba (pip install numba) matplotlib (pip install -U matplotlib) scipy (pip install scipy) ipywidgets (pip install ipywidgets) ipyfilechooser (pip install ipyfilechooser)

codecs (pip install codec) random (pip install random2) re (pip install regex) tkinter (pip install tk)

Input file types:

Ms-T data as no headers temp Ms seperated by comma or space or tab

Thellier (.tdt) files with header seperated by comma, space or tab, no space within in sample names

To use the program:

Click on each cell in the Jupyter notebook following the instructions above the cell

The model uses a full-wdith half maxmiumm method to extrapolate the FORC distribtion to SF = 0. as outlined in Muxworthy and Heslop 2011.

When a noisy FORC diagram is used, anomalous FWHM can be produced and these values can be removed.

The cooling-rate corrected Thellier files are saved as original_file_name_U.tdt in the same folder.
