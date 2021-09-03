# ThelllierCoolPy
A cooling-rate correction tool for Thellier-type paleointensity data. Written in Python.

ThellierCoolPy can be run in Google's Colab or Jupyter. Colab has the advantage that in runs it the cloud and everything is already installed (**no installations needed!**), whilst Jupyter is a little more involved and one of the required packages (numba) currently (August 2021) is not compatable with Apple's Silicon M1 processor released in January 2021. We will update this when the package becomes available.


**Experimental data input file formats**

There are three types of data input:

1) Ms-T data has no headers, simply temperature and Ms seperated by comma, space or tab.
2) Thellier data is in ThellierTool data format (.tdt). That is, files with a header seperated by comma, space or tab, no space within in sample names.
3) FORC data. This is FORC data output format from a Princeton or Lakeshore instrument.

Examples of the input data formats are shown in the example folder.



**Instructions for installation and usage**

**Colab:**

Download all the files from the github Colab folder to a folder of your choice. Run Colab (just search for Colab in browser or enter https://colab.research.google.com/). Colab works best in a Google Chrome browswer. 

To run, use Colab's upload option to load ThelllierTool_colab.ipynb. The Colab version requires that the Thellier data, Ms-T curves and FORC diagram are in the same folder as ThellierCoolFunc_colab.py. These are uploaded to Colab within during processing within the notebook. To run the program, click on the instructions above each cell.  The model uses a full-wdith half maxmiumm method to extrapolate the FORC distribtion to SF = 0, as outlined in Muxworthy and Heslop (2011).  When a noisy FORC diagram is used, anomalous FWHM can be produced and these values can be removed. The cooling-rate corrected Thellier files are saved as original_file_name_U.tdt, and can be downloaded from Colab.



**Jupyter**:

We assume that Jupyter is already installed, if not please use Colab.

To run locally on Jupyter notebook, the following packages need to be installed: 

numpy, numba, matplotlib, scipy, ipywidgets, ipyfileschooser,  random2, regex, tk

First, you may need to upgrade your version of pip using:

python -m pip install --upgrade pip

Second, these packages can be  installed via (just cut and paste):

1. pip install numpy numba scipy ipywidgets ipyfileschooser  random2 regex tk
2. pip install -U matplotlib

Depending on your OS, codecs may be installed as default.

Launch Jupyter, and open ThellierCoolPy_Jupyter.ipynb.  Run each cell (SHIFT + RETURN) following the instructions above each cell. The model uses a full-wdith half maxmiumm method to extrapolate the FORC distribtion to SF = 0, as outlined in Muxworthy and Heslop (2011).  When a noisy FORC diagram is used, anomalous FWHM can be produced and these values can be removed. The cooling-rate corrected Thellier files are saved as original_file_name_U.tdt in the same folder as the original data.


**Further information**

ThellierCoolPy is described in Muxworthy and Baker (2021). If you use ThelllierCoolPy in your research, please cite:

Muxworthy, A.R. and Baker, E.B. (2021), ThellierCoolPy: A cooling-rate correction tool for paleointensity data, submitted to G-Cubed.
