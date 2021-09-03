# ThelllierCoolPy
A cooling-rate correction tool for Thellier-type paleointensity data. Written in Python.

ThellierCoolPy can be run in Google's Colab or Jupyter. Colab has the advantage that in runs in the cloud and everything is already installed (**no installations needed!**), whilst Jupyter is a little more involved and one of the required packages (numba) currently (August 2021) is not compatable with Apple's Silicon M1 processor released in January 2021. We will update this when the package becomes available.

**Experimental data input file formats**

Examples of the input data formats are shown in the example folder:

1) Ms-T data has no headers, simply temperature and Ms seperated by comma, space or tab.
2) Thellier data is in ThellierTool data format (.tdt). That is, files with a header seperated by comma, space or tab, no space within in sample names.
3) FORC data. This is FORC data output format from a Princeton or Lakeshore instrument.


**Instructions for installation and usage**

**Colab:**

Download all the files from the Colab folder. Run Colab (just search for Colab). Colab works best in a Google Chrome browswer. 

To run on the cloud using Google Colab:

The cooling rate correction can also be ran using Google Colab by downloading the colab versions of ThellierCoolPy and ThellierCoolfunc. This requires the Thellier plots, Ms-T curves and FORC diagram to be in the same folder as ThellierCoolFunc_colab to be downloaded onto the cloud for processing within the notebook. 

An example of a colab input folder is the zipped folder: colab_files_example.

The corrected Thellier files can then be downloaded from Google Colab.

**Jupyter** (installed via Anaconda):

Assuming Jupyter is installed. Download 

To run locally on Jupyter notebook, the following packages are required: 

numpy, numba, matplotlib, scipy, ipywidgets, ipyfileschooser, codecs, random2, regex, tk

First, you may need to upgrade your version of pip using:

python -m pip install --upgrade pip

Second, these can be  installed via (just cut and paste):

pip install numpy numba scipy ipywidgets ipyfileschooser codecs random2 regex tk

pip install -U matplotlib

Depending on your OS, codecs maybe installed as default.




**To use the program**

For both Colab and Jupyter, click on each following the instructions above each cell. The model uses a full-wdith half maxmiumm method to extrapolate the FORC distribtion to SF = 0, as outlined in Muxworthy and Heslop (2011).  When a noisy FORC diagram is used, anomalous FWHM can be produced and these values can be removed.

The cooling-rate corrected Thellier files are saved as original_file_name_U.tdt in the same folder.



