#import turbosensei as fs
from numba import jit
import numpy as np
from scipy.linalg import lstsq
from matplotlib.figure import figaspect
from math import acos, pi, degrees, sin, cos, sqrt, log, log10, tanh, remainder
from scipy.interpolate import interp2d
import random
import codecs as cd

import ipywidgets as widgets
from ipywidgets import VBox, HBox
import codecs as cd
import matplotlib.pyplot as plt
import copy as copy
import os, re
from scipy.linalg import lstsq
from numpy.polynomial.polynomial import polyfit
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.interpolate import griddata
from ipyfilechooser import FileChooser

mu0 = 4*pi*1e-7
kb = 1.3806503e-23
tau = 10e-9
roottwohffield = 2**(0.5)  

eglip=0.54
eglid=log(113*((31486**(-eglip))))+0.52
eglid=113*((31486**(-eglip)))
egliin = 5.4
eglip=egliin/10.0 # 0.54 #eglin from data file
hf=(10000**(0.54))*(10**(-0.52)) #checking maths
eglid=+0.52+(log10(hf)-log10(10000.0)*eglip)
affield = 0
affmax = 0.0
af_step_max = 0.0
flatsub=0
flat=0


afone = 1
afzero = 0

#FORC file read 

def FORCfile():
    fc = FileChooser()
    display(fc)
    return fc

def proccess_all(X):

    #4
    #process data
    style = {'description_width': 'initial'} #general style settings
    fn = X['fn'] #print path to md.frc
    sample, unit, mass = sample_details(fn)

    sample_widge = widgets.Text(value=sample,description='Sample name:',style=style) #dont think these do anything
    prop_title = widgets.HTML(value='<h3>Sample preprocessing options</h3>')
    mass_title = widgets.HTML(value='To disable mass normalization use a value of -1')
    if mass == "N/A":
        mass_widge = widgets.FloatText(value=-1, description = 'Sample mass (g):',style=style)
    else:
        mass_widge = widgets.FloatText(value=mass, description = 'Sample mass (g):',style=style)
    mass_widge1 = HBox([mass_widge,mass_title])

    X["sample"] = sample_widge
    X["mass"] = mass_widge
    X["unit"] = unit

    H, Hr, M, Fk, Fj, Ft, dH = parse_measurements(X["fn"])
    Hcal, Mcal, tcal = parse_calibration(X["fn"])
    Hc1, Hc2, Hb1, Hb2 = measurement_limts(X)

    # make a data dictionary for passing large numbers of arguments
        # should unpack in functions for consistency
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["dH"] = dH
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["Ft"] = Ft
    X["Hcal"] = Hcal
    X["Mcal"] = Mcal
    X["tcal"] = tcal
    X["Hc1"] = Hc1
    X["Hc2"] = Hc2
    X["Hb1"] = Hb1
    X["Hb2"] = Hb2
    slope = 70.
    X["slope"] = slope #set value manually for slope correction

    if X['unit']=='Cgs': #mine is SI
        X = CGS2SI(X)
    #print(X["M"])
    X = drift_correction(X) #do drift correction?
    #print(X["M"]) #
    X = slope_correction(X)
    X = remove_fpa(X) #changed value
    #print(X["H"]) #values not changed after drift correction
    X = remove_lpa(X)
    X = lowerbranch_subtract(X)

  #  fig = plt.figure(figsize=(12,8))
   # ax1 = fig.add_subplot(121)
    #X = plot_hysteresis(X,ax1)
   # ax2 = fig.add_subplot(122)
   # X = plot_delta_hysteresis(X,ax2)
    return(X)
    
def prod_FORCs(X):
    maxSF = 5
    X['maxSF1'] = maxSF
    #function to get arrays for plotting
    X = create_arrays(X, maxSF)
    X = nan_values2(X)
    #calculate FORC dsitribution for each SF
    for sf in range(2, maxSF+1):
        X = calc_rho(X, sf)
        sf+=1   
    X = nan_values(X, maxSF) 
    X = rotate_FORC(X)
    return(X)    
    

#### file parsing routines

def parse_header(file,string):
    """Function to extract instrument settings from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    string: instrument setting to be extracted (string)
    Outputs:
    output: value of instrument setting [-1 if setting doesn't exist] (float)
    """
    output=-1 #default output (-1 corresponds to no result, i.e. setting doesn't exist)
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=float(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=float(line[idx+1:])  #value taken as everything to right of the setting name 

    return output


def parse_measurements(file):
    """Function to extract measurement points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    H: Measurement applied field [float, SI units] 
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    Ft: Estimated times at which the points were measured (float, seconds)
    dH: Measurement field spacing [float SI units]
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## determine indicies of each FORC
    idxSAT = np.array(np.where(np.isin(H0, dum))) #find start address of each blank line
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSTART = idxSAT[1::2]+1 #find start address of each FORC
    idxEND = idxSAT[2::2]-1 ##find end address of each FORC

    
    #Extract first FORC to initialize arrays 
    M=M0[idxSTART[0]:idxEND[0]+1] #Magnetization values
    H=H0[idxSTART[0]:idxEND[0]+1] #Field values
    Hr=np.ones(idxEND[0]+1-idxSTART[0])*H0[idxSTART[0]] #Reversal field values
    Fk=np.ones(idxEND[0]+1-idxSTART[0]) #index number of FORC
    Fj=np.arange(1,1+idxEND[0]+1-idxSTART[0])# measurement index within given FORC

    #Extract remaining FORCs one by one into into a long-vector
    for i in range(1,idxSTART.size):
        M=np.concatenate((M,M0[idxSTART[i]:idxEND[i]+1]))
        H=np.concatenate((H,H0[idxSTART[i]:idxEND[i]+1]))
        Hr=np.concatenate((Hr,np.ones(idxEND[i]+1-idxSTART[i])*H0[idxSTART[i]]))
        Fk=np.concatenate((Fk,np.ones(idxEND[i]+1-idxSTART[i])+i))
        Fj=np.concatenate((Fj,np.arange(1,1+idxEND[i]+1-idxSTART[i])))
    
    unit = parse_units(file) #Ensure use of SI units
    
    if unit=='Cgs':
        H=H/1E4 #Convert Oe into T
        Hr=Hr/1E4 #Convert Oe into T
        M=M/1E3 #Convert emu to Am^2

    dH = np.mean(np.diff(H[Fk==np.max(Fk)])) #mean field spacing

    Ft=measurement_times(file,Fk,Fj) #estimated time of each measurement point

    return H, Hr, M, Fk, Fj, Ft, dH
def parse_units(file):
    """Function to extract instrument unit settings ('') from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    CGS [Cgs setting] or SI [Hybrid SI] (string)
    """
    string = 'Units of measure' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idxSI = line.find('Hybrid SI') #will return location if string is found, otherwise returns -1
            idxCGS = line.find('Cgs') #will return location if string is found, otherwise returns -1
    
    if idxSI>idxCGS: #determine which unit string was found in the headerline and output
        return 'SI'
    else:
        return 'Cgs'
def parse_mass(file):
    """Function to extract sample from FORC data file header
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    Mass in g or N/A
    """
    output = 'N/A'
    string = 'Mass' #header definition of units
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in lines_that_start_with(string, fp): #find the line starting with the setting name
            idx = line.find('=') #Some file formats may contain an '='
            if idx>-1.: #if '=' found
                output=(line[idx+1:]) #value taken as everything to right of '='
            else: # '=' not found
                idx = len(string) #length of the setting string 
                output=(line[idx+1:])  #value taken as everything to right of the setting name
        
            if output.find('N/A') > -1:
                output = 'N/A'
            else:
                output = float(output)

    return output
def measurement_times(file,Fk,Fj):
    """Function to estimate the time at which magnetization points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.
    Inputs:
    file: name of data file (string)    
    Fk: FORC indicies (int)
    Fj: Measurement indicies within given FORC
    Outputs:
    Ft: Estimated times at which the magnetization points were measured (float)
    """    
    unit=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if unit=='Cgs':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T

    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH

    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt3 = Hb2/alpha

    Npts=int(Fk.size)
    Ft=np.zeros(Npts)
    
    for i in range(Npts):
        if Fk[i]<=1+nc2:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*(Fk[i]-1)**2
        else:
            Ft[i]=Fk[i]*Dt1+Dt3+Fj[i]*tau+dH/alpha*(Fk[i]*(Fk[i]-1))+(tau-dH/alpha)*((Fk[i]-1)*(1+nc2)-nc2)

    return Ft
def parse_calibration(file):
    """Function to extract measured calibration points from a FORC sequence
    
    Inputs:
    file: name of data file (string)    
    Outputs:
    Hcal: sequence of calibration fields [float, SI units]
    Mcal: sequence of calibration magnetizations [float, SI units]
    tcal: Estimated times at which the calibration points were measured (float, seconds)
    """ 

    dum=-9999.99 #dum value to indicate break in measurement seqence between FORCs and calibration points
    N0=int(1E6) #assume that any file will have less than 1E6 measurements
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: #open the data file (latin9 encoding seems to work, UTF and ASCII don't)
        for line in find_data_lines(fp): #does the current line contain measurement data
            count=count+1 #increase counter
            idx = line.find(',') #no comma indicates a blank linw
            if idx>-1: #line contains a comma
                H0[count]=float(line[0:idx]) #assign field value (1st column)
                line=line[idx+1:] #remove the leading part of the line (only characters after the first comma remain)
                idx = line.find(',') #find next comman
                if idx>-1: #comma found in line
                    M0[count]=float(line[0:idx]) #read values up to next comma (assumes 2nd column is magnetizations)
                else: #comma wasn't found   
                    M0[count]=float(line) # magnetization value is just the remainder of the line 
            else:
                H0[count]=dum #line is blank, so fill with dummy value
                M0[count]=dum #line is blank, so fill with dummy value

    idx_start=np.argmax(H0!=dum) #find the first line that contains data            
    M0=M0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector           
    M0=M0[~np.isnan(M0)] #remove any NaNs at the end of the array
    H0=H0[idx_start-1:-1] #strip out leading dummy values from magnetizations, leaving 1 dummy at start of vector
    H0=H0[~np.isnan(H0)] #remove any NaNs at the end of the array

    ## now need to pull out the calibration points, will be after alternate -9999.99 entries
    idxSAT = np.array(np.where(np.isin(H0, dum))) #location of dummy values
    idxSAT = np.ndarray.squeeze(idxSAT) #squeeze into 1D
    idxSAT = idxSAT[0::2]+1 #every second index+1 should be calibration points

    Hcal=H0[idxSAT[0:-1]] #calibration fields
    Mcal=M0[idxSAT[0:-1]] #calibration magnetizations
    tcal=calibration_times(file,Hcal.size) #estimate the time of each calibratio measurement

    unit = parse_units(file)
    
    if unit=='Cgs': #ensure SI units
        Hcal=Hcal/1E4 #convert from Oe to T
        Mcal=Mcal/1E3 #convert from emu to Am^2

    return Hcal, Mcal, tcal
def calibration_times(file, Npts):
    """Function to estimate the time at which calibration points were measured in a FORC sequence
    
    Follows the procedure given in:
    R. Egli (2013) VARIFORC: An optimized protocol for calculating non-regular first-order reversal curve (FORC) diagrams. Global and Planetary Change, 110, 302-320, doi:10.1016/j.gloplacha.2013.08.003.
    Inputs:
    file: name of data file (string)    
    Npts: number of calibration points (int)
    Outputs:
    tcal_k: Estimated times at which the calibration points were measured (float)
    """    
    unit=parse_units(file) #determine measurement system (CGS or SI)

    string='PauseRvrsl' #Pause at reversal field (new file format, -1 if not available)
    tr0=parse_header(file,string)
    
    string='PauseNtl' #Pause at reversal field (old file format, -1 if not available)
    tr1=parse_header(file,string)

    tr=np.max((tr0,tr1)) #select Pause value depending on file format
    
    string='Averaging time' #Measurement averaging time 
    tau=parse_header(file,string)

    string='PauseCal' #Pause at calibration point
    tcal=parse_header(file,string)

    string='PauseSat' #Pause at saturation field
    ts=parse_header(file,string)

    string='SlewRate' #Field slewrate
    alpha=parse_header(file,string)

    string='HSat' #Satuation field
    Hs=parse_header(file,string)

    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(file,string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(file,string)

    string='Hc2' #upper Hc value for the FORC box (n.b. Hc1 is assumed to be 0)
    Hc2=parse_header(file,string)

    string='NForc' # Numer of measured FORCs (new file format, -1 if not available)
    N0=parse_header(file,string)

    string='NCrv'  # Numer of measured FORCs (old file format, -1 if not available)
    N1=parse_header(file,string)

    N=np.max((N0,N1)) #select Number of FORCs depending on file format

    if unit=='Cgs':
        alpha=alpha/1E4 #convert from Oe to T
        Hs=Hs/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T
    
    dH = (Hc2-Hb1+Hb2)/N #estimated field spacing
    
    #now following Elgi's estimate of the measurement time
    nc2 = Hc2/dH
    Dt1 = tr + tau + tcal + ts + 2.*(Hs-Hb2-dH)/alpha
    Dt2 = tr + tau + (Hc2-Hb2-dH)/alpha

    Npts=int(Npts)
    tcal_k=np.zeros(Npts)
    
    for k in range(1,Npts+1):
        if k<=1+nc2:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*(k-1)**2
        else:
            tcal_k[k-1]=k*Dt1-Dt2+dH/alpha*k**2+(tau-dH/alpha)*((k-1)*(1+nc2)-nc2)

    return tcal_k
def sample_details(fn):

    sample = fn.split('/')[-1]
    sample = sample.split('.')
    
    if type(sample) is list:
        sample=sample[0]

    units=parse_units(fn)
    mass=parse_mass(fn)
  
    return sample, units, mass
def measurement_limts(X):
    """Function to find measurement limits and conver units if required
    Inputs:
    file: name of data file (string)    
    Outputs:
    Hc1: minimum Hc
    Hc2: maximum Hc
    Hb1: minimum Hb
    Hb2: maximum Hb
    """    
    
    string='Hb2' #upper Hb value for the FORC box
    Hb2=parse_header(X["fn"],string)

    string='Hb1' #lower Hb value for the FORC box
    Hb1=parse_header(X["fn"],string)

    string='Hc2' #upper Hc value for the FORC box
    Hc2=parse_header(X["fn"],string)

    string='Hc1' #lower Hc value for the FORC box
    Hc1=parse_header(X["fn"],string)

    if X['unit']=='Cgs': #convert CGS to SI
        Hc2=Hc2/1E4 #convert from Oe to T
        Hc1=Hc1/1E4 #convert from Oe to T
        Hb2=Hb2/1E4 #convert from Oe to T
        Hb1=Hb1/1E4 #convert from Oe to T  

    return Hc1, Hc2, Hb1, Hb2

#### Unit conversion ####
def CGS2SI(X):
    
    X["H"] = X["H"]/1E4 #convert Oe into T
    X["M"] = X["M"]/1E3 #convert emu to Am2
      
    return X

#### low-level IO routines
def find_data_lines(fp):
    """Helper function to identify measurement lines in a FORC data file.
    
    Given the various FORC file formats, measurements lines are considered to be those which:
    Start with a '+' or,
    Start with a '-' or,
    Are blank (i.e. lines between FORCs and calibration points) or,
    Contain a ','
    Inputs:
    fp: file identifier
    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if ((line.startswith('+')) or (line.startswith('-')) or (line.strip()=='') or line.find(',')>-1.)]
def lines_that_start_with(string, fp):
    """Helper function to lines in a FORC data file that start with a given string
    
    Inputs:
    string: string to compare lines to 
    fp: file identifier
    Outputs:
    line: string corresponding to data line that meets the above conditions
    """
    return [line for line in fp if line.startswith(string)]


#### PREPROCESSING OPTIONS ####
def options(X):
    style = {'description_width': 'initial'} #general style settings

    ### Define sample properties ###
    fn = X['fn']
    prop_title = widgets.HTML(value='<h3>Sample preprocessing options</h3>')
    mass_title = widgets.HTML(value='To disable mass normalization use a value of -1')

    sample, unit, mass = ut.sample_details(fn)

    sample_widge = widgets.Text(value=sample,description='Sample name:',style=style)
    
    if mass == "N/A":
        mass_widge = widgets.FloatText(value=-1, description = 'Sample mass (g):',style=style)
    else:
        mass_widge = widgets.FloatText(value=mass, description = 'Sample mass (g):',style=style)

    mass_widge1 = HBox([mass_widge,mass_title])
    
    ### Define measurement corrections ###
    correct_title = widgets.HTML(value='<h3>Select preprocessing options:</h3>')
    
    slope_widge = widgets.FloatSlider(
        value=70,
        min=1,
        max=100.0,
        step=1,
        description='Slope correction [%]:',
        style=style,
        readout_format='.0f',
    )
    
    slope_title = widgets.HTML(value='To disable high-field slope correction use a value of 100%')
    slope_widge1 = HBox([slope_widge,slope_title])
    
    drift_widge = widgets.Checkbox(value=False, description='Measurement drift correction')
    fpa_widge = widgets.Checkbox(value=False, description='Remove first point artifact')
    lpa_widge = widgets.Checkbox(value=False, description='Remove last point artifact')
    correct_widge = VBox([correct_title,sample_widge,mass_widge1,slope_widge1,drift_widge,fpa_widge,lpa_widge])

    preprocess_nest = widgets.Tab()
    preprocess_nest.children = [correct_widge]
    preprocess_nest.set_title(0, 'PREPROCESSING')
    display(preprocess_nest)

    X["sample"] = sample_widge
    X["mass"] = mass_widge
    X["unit"] = unit
    X["drift"] = drift_widge
    X["slope"] = slope_widge
    X["fpa"] = fpa_widge
    X["lpa"] = lpa_widge
    
    return X

#### PREPROCESSING COMMAND ####
def execute(X):
  
    #parse measurements
    H, Hr, M, Fk, Fj, Ft, dH = ut.parse_measurements(X["fn"])
    Hcal, Mcal, tcal = ut.parse_calibration(X["fn"])
    Hc1, Hc2, Hb1, Hb2 = ut.measurement_limts(X)
    
    # make a data dictionary for passing large numbers of arguments
    # should unpack in functions for consistency
    X["H"] = H
    X["Hr"] = Hr
    X["M"] = M
    X["dH"] = dH
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["Ft"] = Ft
    X["Hcal"] = Hcal
    X["Mcal"] = Mcal
    X["tcal"] = tcal
    X["Hc1"] = Hc1
    X["Hc2"] = Hc2
    X["Hb1"] = Hb1
    X["Hb2"] = Hb2

    if X['unit']=='Cgs':
        X = ut.CGS2SI(X)
    
    if X["drift"].value == True:
        X = drift_correction(X)   
  
    if X["slope"].value < 100:
        X = slope_correction(X)
  
    if X["fpa"].value == True:
        X = remove_fpa(X)
    
    if X["lpa"].value == True:
        X = remove_lpa(X)
    
    #extend FORCs
    X = FORC_extend(X)

    #perform lower branch subtraction
    X = lowerbranch_subtract(X)
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(121)
    X = plot_hysteresis(X,ax1)
    ax2 = fig.add_subplot(122)
    X = plot_delta_hysteresis(X,ax2)
    
    outputfile = X["sample"].value+'_HYS.eps'    
    plt.savefig(outputfile, bbox_inches="tight")
    plt.show()
    
    return X

#### PREPROCESSING ROUTINES ####
def remove_lpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Ft = X["Ft"]
    
    #remove last point artifact
    Nforc = int(np.max(Fk))
    W = np.ones(Fk.size)
    
    for i in range(Nforc):      
        Fj_max=np.sum((Fk==i))
        idx = ((Fk==i) & (Fj==Fj_max))
        W[idx]=0.0
    
    idx = (W > 0.5)
    H=H[idx]
    Hr=Hr[idx]
    M=M[idx]
    Fk=Fk[idx]
    Fj=Fj[idx]
    Ft=Ft[idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def remove_fpa(X):
    
    #unpack
    Fj = X["Fj"]
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    Ft = X["Ft"]
    
    #remove first point artifact
    idx=((Fj==1.0))
    H=H[~idx]
    Hr=Hr[~idx]
    M=M[~idx]
    Fk=Fk[~idx]
    Fj=Fj[~idx]
    Ft=Ft[~idx]
    Fk=Fk-np.min(Fk)+1. #reset FORC number if required
    Fj=Fj-1.
    
    #repack
    X["Fj"] = Fj
    X["H"] = H   
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Ft"] = Ft        
    
    return X

def drift_correction(X):
  
    #unpack
    M = X["M"]
    Mcal = X["Mcal"]    
    Ft = X["Ft"]
    tcal = X["tcal"]
  
    #perform drift correction
    M=M*Mcal[0]/np.interp(Ft,tcal,Mcal,left=np.nan) #drift correction
  
    #repack
    X["M"] = M
  
    return X

def FORC_extend(X):
    
    Ne = 20 #extend up to 20 measurement points backwards
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    for i in range(int(X['Fk'][-1])):
        M0 = M[Fk==i+1]
        H0 = H[Fk==i+1]
        Hr0 = Hr[Fk==i+1][0]
        
        M1 = M0[0] - (np.flip(M0)[1:]-M0[0])
        H1 = H0[0] - (np.flip(H0)[1:]-H0[0])
            
        if M1.size>Ne:
            H1 = H1[-Ne-1:-1]
            M1 = M1[-Ne-1:-1]
        
        if i==0:    
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H1,H0))
            M_new = np.concatenate((M1,M0))
            Hr_new = np.ones(N_new)*Hr0
            Fk_new = np.ones(N_new)
            Fj_new = np.arange(N_new)+1-M1.size
        else:
            N_new = np.concatenate((M1,M0)).size
            H_new = np.concatenate((H_new,H1,H0))
            M_new = np.concatenate((M_new,M1,M0))
            Hr_new = np.concatenate((Hr_new,np.ones(N_new)*Hr0))
            Fk_new = np.concatenate((Fk_new,np.ones(N_new)+i))
            Fj_new = np.concatenate((Fj_new,np.arange(N_new)+1-M1.size))
            
    #pack up variables
    X['H'] = H_new
    X['Hr'] = Hr_new
    X['M'] = M_new
    X['Fk'] = Fk_new
    X['Fj'] = Fj_new
    
    return X

def lowerbranch_subtract(X):
    """Function to subtract lower hysteresis branch from FORC magnetizations
    
    Inputs:
    H: Measurement applied field [float, SI units]
    Hr: Reversal field [float, SI units]
    M: Measured magnetization [float, SI units]
    Fk: Index of measured FORC (int)
    Fj: Index of given measurement within a given FORC (int)
    
    Outputs:
    M: lower branch subtracted magnetization [float, SI units]
   
    
    """
    
    #unpack
    H = X["H"]    
    Hr = X["Hr"]
    M = X["M"]
    Fk = X["Fk"]
    Fj = X["Fj"]
    dH = X["dH"]
    
    Hmin = np.min(H)
    Hmax = np.max(H)


    Nbar = 10
    nH = int((Hmax - Hmin)/dH)
    Hi = np.linspace(Hmin,Hmax,nH*50+1)
    Mi = np.empty(Hi.size)
    
    #perform basic loess
    for i in range(Hi.size):
        idx = (H>=Hi[i]-2.5*dH) & (H<=Hi[i]+2.5*dH)
        Mbar = M[idx]
        Hbar = H[idx]
        Fbar = Fk[idx]
        F0 = np.sort(np.unique(Fbar))
        if F0.size>Nbar:
            F0=F0[-Nbar]
        else:
            F0=np.min(F0)
        idx = Fbar>=F0
        
        p = np.polyfit(Hbar[idx],Mbar[idx],2)
        Mi[i] = np.polyval(p,Hi[i])
    
    Hlower = Hi
    Mlower = Mi
    Mcorr=M-np.interp(H,Hlower,Mlower,left=np.nan,right=np.nan) #subtracted lower branch from FORCs via interpolation

    Fk=Fk[~np.isnan(Mcorr)] #remove any nan
    Fj=Fj[~np.isnan(Mcorr)] #remove any nan
    H=H[~np.isnan(Mcorr)] #remove any nan
    Hr=Hr[~np.isnan(Mcorr)] #remove any nan
    M=M[~np.isnan(Mcorr)] #remove any nan
    Mcorr = Mcorr[~np.isnan(Mcorr)] #remove any nan
    
    #repack
    X["H"] = H    
    X["Hr"] = Hr
    X["M"] = M
    X["Fk"] = Fk
    X["Fj"] = Fj
    X["DM"] = Mcorr
    
    return X

    ###### HELPER FUNCTIONS TO READ FROM FILE

def slope_correction(X):
  
    #unpack
    H = X["H"]
    M = X["M"]
  
    # high field slope correction
    Hidx = H > (X["slope"]/100) * np.max(H)
    p = np.polyfit(H[Hidx],M[Hidx],1)
    M = M - H*p[0]
  
    #repack
    X["M"]=M
  
    return X



#### PLOTTING ROUTINES #####

def plot_hysteresis(X,ax):

  #unpack 
    M = X["M"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    #mpl.style.use('seaborn-whitegrid')
    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k')        

    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim,ylim])
  
    #ax.set_ylim([-1,1])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()
    xmax = np.max(np.abs(ax.get_xlim()))
    ax.set_xlim([-xmax,xmax])

    #label x-axis
    ax.set_xlabel('$\mu_0 H [T]$',horizontalalignment='right', position=(1,25), fontsize=12)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        ax.set_ylabel('$M [Am^2/kg]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
    else: 
        ax.set_ylabel('$M [Am^2]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)

    
    X["xmax"]=xmax
    
    return X

def plot_delta_hysteresis(X,ax):

    #unpack 
    M = X["DM"]
    H = X["H"]
    Fk = X["Fk"]
    Fj = X["Fj"]

    hfont = {'fontname':'STIXGeneral'}

    for i in range(5,int(np.max(Fk)),5):
    
        if X["mass"].value > 0.0: #SI and mass normalized (T and Am2/kg)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)]/(X["mass"].value/1000.0),'-k')        
        else: #SI not mass normalized (T and Am2)
            ax.plot(H[(Fk==i) & (Fj>0)],M[(Fk==i) & (Fj>0)],'-k') 
      
    ax.grid(False)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='major',direction='out',length=5,width=1,labelsize=12,color='k')
    ax.tick_params(axis='both',which='minor',direction='out',length=5,width=1,color='k')

    ax.spines['left'].set_position('zero')
    ax.spines['left'].set_color('k')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()
  
    ylim=np.max(np.abs(ax.get_ylim()))
    ax.set_ylim([-ylim*0.1,ylim])
    yticks0 = ax.get_yticks()
    yticks = yticks0[yticks0 != 0]
    ax.set_yticks(yticks)
  
    # set the y-spine
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('k')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    Xticks = ax.get_xticks()
    Xidx = np.argwhere(np.abs(Xticks)>0.01)
    ax.set_xticks(Xticks[Xidx])

    xmax = X["xmax"]
    ax.set_xlim([-xmax,xmax])
    
    #label x-axis according to unit system
    ax.set_xlabel('$\mu_0 H [T]$',horizontalalignment='right', position=(1,25), fontsize=12)

    #label y-axis according to unit system
    if X["mass"].value > 0.0:
        ax.set_ylabel('$M - M_{hys} [Am^2/kg]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)
    else: 
        ax.set_ylabel('$M - M_{hys} [Am^2]$',verticalalignment='top',position=(25,0.9), fontsize=12,**hfont)

    
    return X

def create_arrays(X, maxSF):

    Fk_int = (X['Fk'].astype(int)) #turn to int to use bincount
    counts = np.bincount(Fk_int) #time each no. appears in Fk (no. FORC on)
    max_FORC_len = np.max(counts) #max occurance of a FORC no. = longest FORC length = no. columns
    no_FORC = np.argmax(counts) #max FORC no.   = rows

    H_A = np.zeros((no_FORC, max_FORC_len)) #initialize arrays
    Hr_A = np.zeros((no_FORC, max_FORC_len))
    M_A = np.zeros((no_FORC, max_FORC_len))
    Fk_A = np.zeros((no_FORC, max_FORC_len))
    Rho = np.zeros((maxSF+1, no_FORC, max_FORC_len))
    #initialize zero values
    H_A[0,0] = X['H'][0]
    Hr_A[0,0] = X['Hr'][0]
    M_A[0,0] = X['M'][0]
    Fk_A[0,0] = X['Fk'][0]

    j=0 # just filled first point in first row
    i=0 # start at first row
    for cnt in range(1,len(X['Fk']+1)):
        if (X['Fk'][cnt] == X['Fk'][cnt-1]): #if Fk no is the same, stay on same row and fill data
            j +=1 #add one more to column and repeat
            H_A[i][j] = X['H'][cnt]
            Hr_A[i][j] = X['Hr'][cnt]
            M_A[i][j] = X['M'][cnt]     
        else:
            i +=1 #new row
            j = 0 #set column index back to zero
            H_A[i][j] = X['H'][cnt]
            Hr_A[i][j] = X['Hr'][cnt]
            M_A[i][j] = X['M'][cnt]            
        cnt +=1 #next point
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A
    X['rho'] = Rho
    X['no_FORC'] = no_FORC
    X['max_FORC_len'] = max_FORC_len
    return(X)


def nan_values2(X):
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    M_A = X['M_A']
    for i in range(len(H_A)):
        for j in range(len(Hr_A[0])):
            if (H_A[i][j] == 0.0):
                H_A[i][j] = 'NaN'
                Hr_A[i][j] = 'NaN'
                M_A[i][j] = 'NaN'
                

    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A
    return(X)

def calc_rho(X, SF):
    no_FORC = X['no_FORC']
    max_FORC_len = X['max_FORC_len']
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    M_A = X['M_A']
    Rho = X['rho']

    for i in range(no_FORC): #find main points, test without +1
        for j in range(max_FORC_len): #find each j indice
            #locate smoothing grids

            cnt = 0
            h1 = min(i, SF) #always on row go from SF below and SF above. no diffrence if this is i-1 and in k1 (j-1)
            h2 = min(SF, (no_FORC - i)) #try and loop over all points and not ignore boundaries,
            k1 = min(j, SF) #point to left, 1j if 0 etc or SF is in middle
            k2 = min(SF, (max_FORC_len-j)) #right hand side - either SF or if near edge do total - j (point at)

            A = np.zeros(((h2+h1+1)*(k1+k2+1),6))
            b = np.zeros(((h2+h1+1)*(k1+k2+1)))
            A[:,:] = np.nan
            b[:] = np.nan

            #if (M_A[i][j] != 0. and H_A[i][j] !=0 and Hr_A[i][j] != 0): 
            if (H_A[i][j] > Hr_A[i][j]):
                for h in range((-h1), (h2+1)): #loop over row in smoothing window
                    for k in range((-k1), (k2+1)): #loop over columns in smoothing window
                        if ((j+h+k) >= 0 and (j+k+h) < (max_FORC_len) and (i+h) >= 0 and (i+h) < (no_FORC)): 
                                #if (M_A[i+h][j+h+k] != 0. and H_A[i+h][j+h+k] !=0 and Hr_A[i][j+h+k] != 0): #remved but this makes a difference to plot
                            A[cnt, 0] = 1.
                            A[cnt, 1] = Hr_A[i+h][j+k+h] - Hr_A[i][j]
                            A[cnt, 2] = (Hr_A[i+h][j+k+h] - Hr_A[i][j])**2.
                            A[cnt, 3] = H_A[i+h][j+k+h] - H_A[i][j]
                            A[cnt, 4] = (H_A[i+h][j+k+h] - H_A[i][j])**2.
                            A[cnt, 5] = (Hr_A[i+h][j+k+h] - Hr_A[i][j])*(H_A[i+h][j+k+h] - H_A[i][j])
                            b[cnt] = M_A[i+h][j+k+h]

                            cnt+=1 #count number values looped over
                    #print('A', A)
                    #print('b', b)
                A = A[~np.isnan(A).any(axis=1)]
                b = b[~np.isnan(b)]
                    #print('A no nan', A)
                    #print('b no nan', b)
                if (len(A)>=2): #min no. points to need to smooth over
                        #cmatrix = np.matmul(np.transpose(A), A)
                    dmatrix, res, rank, s = lstsq(A,b)
                    Rho[SF][i][j] = (-1.*(dmatrix[5]))/2.

                else:
                    Rho[SF][i][j] = 0.
            else:
                Rho[SF][i][j] = 0.
            j +=1
        i += 1

    X['H_A'] = H_A #repack variables
    X['Hr_A'] = Hr_A
    X['M_A'] = M_A
    X['rho'] = Rho
    X['no_FORC'] = no_FORC
    X['max_FORC_len'] = max_FORC_len
    return(X)
    
    
def nan_values(X, maxSF):
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    Rho = X['rho']
    for i in range(len(H_A)):
        for j in range(len(Hr_A[0])):
            if (H_A[i][j] == 0.0):
                H_A[i][j] = 'NaN'
                Hr_A[i][j] = 'NaN'
                
    for k in range(maxSF+1):
        for i in range(len(H_A)):
            for j in range(len(Hr_A[0])):
                if (Rho[k][i][j] == 0.0):
                    Rho[k][i][j] = 'NaN'
    X['H_A'] = H_A
    X['Hr_A'] = Hr_A
    X['rho'] = Rho
    return(X)
    
  
def rotate_FORC(X):
    H_A = X['H_A']
    Hr_A = X['Hr_A']
    Hc= (H_A - Hr_A)/2. #x axis
    Hu = (H_A + Hr_A)/2. #y axis
    X['Hc'] = Hc
    X['Hu'] = Hu
    return(X)  
    


    
def norm_rho_all(X):
    Rho = X['rho']
    Rho_n = np.copy(Rho)
    a, x, y = np.shape(Rho)
    X['max_Rho'] = np.zeros((a))
   
    k=0
    for k in range(2,a):
        i = 0
        j = 0
        max_Rho = np.nanmax(Rho_n[k])

        for i in range(x):
            for j in range(y):
                Rho_n[k][i][j] = Rho[k][i][j]/max_Rho
        X['rho_n'] = Rho_n
        X['max_Rho'][k] = max_Rho
        k+=1
    return(X)    

  
def plot_sample_FORC(x, y, z, SF, sample_name):
    z = z[SF]
    zn = np.copy(z)
    # zp = z/(np.nanmax(zn))
    xp = x*1000
    yp = y*1000
    # print(sample_name)
    con = np.linspace(0.1, 1, 9)


    #need to edit labels from input
    cmap, vmin, vmax = FORCinel_colormap(zn) #runs FORCinel colormap

    plt.contourf(xp, yp, zn, 50, cmap= cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(ticks=[0,0.2, 0.4, 0.6, 0.8, 1], format = '%.1f')
    plt.contour(xp, yp, zn, con, colors = 'k')
    plt.xlabel('$\mathrm{h_{c}}$ (mT)', fontsize=14)
    plt.ylabel('$\mathrm{h_{s}}$ (mT)', fontsize=14)

    plt.xlim(0, np.nanmax(xp))
    plt.ylim(np.nanmin(yp), np.nanmax(yp))
    plt.tick_params(axis='both', which='major', labelsize=12)

    cbar.ax.tick_params(labelsize=12)
  #  cbar.ax.set_yticklabels(np.arange(0, 1, 5), fontsize=16, weight='bold')

    plt.title('{} FORC diagram'.format(sample_name))
    #plt.title("Normalised FORC diagram for sample '{0}', using smoothing factor '{1}'".format(sample_name, SF))
    plt.tight_layout()
    plt.show
  #  plt.savefig('Unbounded_FORC_diagram_{}.pdf'.format(X['name'], bbox_inches='tight')
    return     
    
def plot_sample_FORC2(x, y, z, SF, sample_name, xm, ym2):
    z = z[SF]
    zn = np.copy(z)
    
    xp = x*1000
    yp = y*1000
    # print(sample_name)
    con = np.linspace(0.1, 1, 9)
   

    #need to edit labels from input
    cmap, vmin, vmax = FORCinel_colormap(zn) #runs FORCinel colormap
    
    #FORCinel_colormap(z)
    plt.contourf(xp, yp, zn, 50, cmap= cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(ticks=[0,0.2, 0.4, 0.6, 0.8, 1], format = '%.1f')
    plt.contour(xp, yp, zn, con, colors = 'k')
    plt.xlabel('$\mathrm{h_{c}}$ (mT)', fontsize=14)
    plt.ylabel('$\mathrm{h_{s}}$ (mT)', fontsize=14)
    plt.xlim(0, xm)
    plt.ylim(-ym2, ym2)

    plt.tick_params(axis='both', which='major', labelsize=12)

    cbar.ax.tick_params(labelsize=12)
    plt.title('FORC diagram for sample {}, SF = {}'.format(sample_name, SF))
    plt.savefig('FORC_diagram_sample_{}_SF_{}.pdf'.format(sample_name,SF), bbox_inches='tight')
    plt.close()
    return         
    

   
def finding_fwhm(X):
    maxSF = 5
    SFlist = []
    fwhmlist = []
    name = X['name']
    X['fwhmlist'] = fwhmlist
    for SF in range(2,maxSF+1):
        SFlist.append(SF)
    X['SF_list'] = SFlist    
    for i in range(len(SFlist)):
        #print('SF', SFlist[i])
        X = find_fwhm(X, SFlist[i], name)
        i+=1 #0,1,2,3    
    return (X)
    
#new FWHM function
def half_max_test(fwHu_c, fwRho_c, ym):
    arr_L = np.where(fwRho_c == ym)[0]
    L = arr_L[0]
    half_ym = ym/2. #half max
    b = L+1

    while (b < len(fwRho_c)): #stop getting stuck in array

        if(fwRho_c[b] < half_ym):
            
            break
        b = b + 1
    
    top = fwRho_c[b-1] - fwRho_c[b]
    bot = fwHu_c[b-1] - fwHu_c[b]
   
    mo_test = top/bot
    r0 = fwHu_c[b] + ((half_ym - fwRho_c[b])/mo_test)
   
    u = L-1

    while (u > 0): #stop getting stuck in array
       
        if (fwRho_c[u] < half_ym):
            
            break
        u = u - 1
    

    m1 = (fwRho_c[u] - fwRho_c[u+1])/(fwHu_c[u] - fwHu_c[u+1])

    r1 = fwHu_c[u+1] + ((half_ym - fwRho_c[u+1])/m1)
  
    fwhm = r1 - r0
   
    return fwhm, r0, r1

#@jit
def find_fwhm(X, SF, sample_name): #do for 1 SF - 
    fwhmlist = X['fwhmlist']
    Rho = X['rho'] #in functins
    #print('Rho', Rho)
    Hu = X['Hu']
    #print('Hu', Hu)
    #poss add in loop 
    indices = np.unravel_index(np.nanargmax(Rho[SF]),Rho[SF].shape)
    #print('indices', indices)
    fwHu = []
    fwRho = []
    for i in range(len(Rho[SF])):
        fwHu.append(Hu[i][indices[1]]) #add in SF
        fwRho.append(Rho[SF][i][indices[1]])
        i+=1

    fwHu = np.array(fwHu)
    fwRho = np.array(fwRho)
    fwHu = fwHu[~np.isnan(fwHu)]
    fwRho = fwRho[~np.isnan(fwRho)] #have my arrays for fwhm calc
    r0 = 1
    r1 = -1

    
    #here adjust size 
    loc_o = np.argmin(abs(fwHu))
    fwHu_f = fwHu[:loc_o] #loc zero needed
    fwRho_f = fwRho[:loc_o] #loc zero needed

    loc_m = np.argmin(abs(fwRho_f))
    fwHu_c = fwHu[loc_m:(loc_o +(loc_o - loc_m))] #loc zero needed
    fwRho_c = fwRho[loc_m:(loc_o +(loc_o - loc_m))] #loc zero needed

    plt.plot(fwHu_c, fwRho_c)
    plt.show
    m_rho_a = np.sort(fwRho_c)
    i = 1
    while ((r0 >0) or (r1 < 0)): #opposte to FWHM crossing 0 
        ym = m_rho_a[-i]
        #print('i', i, 'max rho in zone', ym)
        # find the two crossing points
        try:
           # hmx = half_max_x(fwHu_c,fwRho_c, ym)
            fwhm, r0, r1 = half_max_test(fwHu_c, fwRho_c, ym)
            #r0 = hmx[0]
            #r1 = hmx[1]
           # print('r0', r0, 'r1', r1)
            #fwhm = r0 - r1
        except:
            print('Error in calculating FWHM for SF %',SF)
            pass
        
        if (i >5):
            print('SF {} is too noisy'.format(SF))
            fwhm = 'Nan'
            r0 = 'NaN'
            r1 = 'NaN'
            break
        i+=1
  
    fwhmlist.append(fwhm)

    half = max(fwRho_c)/2.0
    plt.plot([r0, r1], [half, half], label = SF)
    plt.xlabel('Hu')
    plt.ylabel('FORC weighting')
    plt.legend()
    
    plt.title('Plot of the cross sections of the FWHM')
    #plt.savefig('fwhm_graph.pdf')
    plt.show
    X['fwhmlist'] = fwhmlist
    return(X)
    

def plot_fwhm_1(X):
    SFlist = X['SF_list']
    fwhmlist = X['fwhmlist']
    st_line_SFlist = []
    polyfwhm = []
    polySF = []
    #print(X['maxSF1'])
    maxSF1 = X['maxSF1']
    for i in range(maxSF1+1):
        st_line_SFlist.append(i)
        i +=1

    st_line_SFlist= np.array(st_line_SFlist)
    SFlist = np.array(SFlist)
   # fwhmlist = np.array(fwhmlist)
   # print('fwhmlist 1', fwhmlist)
    for i in range(len(fwhmlist)):
       if (fwhmlist[i] != 'Nan') and (fwhmlist[i] != 'NaN'): #add in fwhmlist[i] != nan
            polyfwhm.append(float(fwhmlist[i]))
            polySF.append(float(SFlist[i]))

    plt.scatter(polySF, polyfwhm)
   # plt.xlim(0,5.3)
   # plt.ylim(0, 0.045)

    b, m = polyfit(polySF, polyfwhm, 1)
    X['b'] = b
    plt.title('FWHM plot with all points')
    plt.xlabel('SF')
    plt.ylabel('FWHM')
    plt.plot(st_line_SFlist, b + m * st_line_SFlist, '-')
    #plt.savefig('test_fwhm.pdf')
    plt.show 
    #plt.savefig('test_fwhm.pdf') 
    Hu = X['Hu']

    i=0

    X['fwhmlist'] = fwhmlist
    #X['sf_correct'] = sf_correct #switch back to 2
    #X['Hu_0'] = Hu_0
    return(X)
    

def plot_fwhm(X):
    SFlist = X['SF_list']
    fwhmlist = X['fwhmlist']
    st_line_SFlist = []
    polyfwhm = []
    polySF = []
    #print('max SF', X['maxSF1'])
    maxSF1 = X['maxSF1']
    for i in range(maxSF1+1):
        st_line_SFlist.append(i)
        i +=1

    st_line_SFlist= np.array(st_line_SFlist)
   # print('stlinesflist', st_line_SFlist)
    SFlist = np.array(SFlist)
   # print('SFlist', SFlist)
    #fwhmlist = np.array(fwhmlist)
    #print('fwhmlist', fwhmlist)

    for i in range(len(SFlist)): #change from fwhmlist to SFlist
        if (fwhmlist[i] != 'che') and (fwhmlist[i] != 'Nan'): #remove value replacement
     #       print('value in loop', fwhmlist[i])
      #      print('type', type(fwhmlist[i]))
            polyfwhm.append(float(fwhmlist[i]))
            polySF.append(float(SFlist[i]))

    plt.scatter(polySF, polyfwhm)
   # plt.xlim(0,5.3)
   # plt.ylim(0, 0.045)
   # print('polySF', polySF)
   # print('polyfwhm', polyfwhm)
    b, m = polyfit(polySF, polyfwhm, 1)
    X['b'] = b
    plt.xlabel('SF')
    plt.ylabel('FWHM')
    plt.title('FWHm plot with accepted points')
    plt.plot(st_line_SFlist, b + m * st_line_SFlist, '-')
    #plt.savefig('test_fwhm.pdf')
    plt.show 
   # plt.savefig('test_fwhm.pdf') 
    Hu = X['Hu']

    i=0
    for i in range(len(SFlist)):
       # print(' 1528 fwhmlist[i]', fwhmlist[i])
        if (fwhmlist[i] == 'Nan') or (fwhmlist[i] == 'che'):
        #    print('edit fwhmlist', fwhmlist[i])
            fwhmlist[i] = float(m*SFlist[i] + b)

    fwhmlist = np.array(fwhmlist)
    fwhmlist = fwhmlist.astype(np.float) 
    X['fwhmlist'] = fwhmlist
#
    #print(maxSF)
 #   while True:
  #      sf_choose = (input("Pick a SF between 2 and 5 to calculaute Sf=0 from:" ))
   #   #  print(type(sf_choose))
    #    try:
     #       sf_choose = int(sf_choose)
      #      if (sf_choose >= 2) and (sf_choose <= maxSF1):
       #         print('in bounds')
       #         break
       # except ValueError:
       #     print('Not an interger')
       #     True
       # if (isinstance(sf_choose, int)):
       # print('int')
  
 

    #X['sf_choose'] = sf_choose
    
   # Hu_0 = Hu*(b/fwhmlist[sf_choose-2])
   # sf_correct = (b/fwhmlist[sf_choose-2])
    X['fwhmlist'] = fwhmlist
    #X['sf_correct'] = sf_correct #switch back to 2
    #X['Hu_0'] = Hu_0
    return(X)
    
    
def check_fwhm(X):
    SFlist = X['SF_list']
    answer = None
    answer2 = None
    fwhmlist = X['fwhmlist']
    maxSF1 = X['maxSF1']
  
    while answer not in ("yes", "no"):
        answer = input("Are any the FWHM unreliable? Enter yes or no: ")
        if (answer == "yes"):
             #rest of code
            while True:
                sf_pick = (input("Which SF is unrealiable and needs to be removed?:" ))
               # print(type(sf_pick))
                while answer2 not in ("yes", "no"):
                    answer2 = input("Are any other FWHM unreliable? Enter yes or no: ")
                    print(answer2)
                    if (answer2 == "yes"):
                        print(answer2)
                        k=1
                        while ( k != 0):
                            try:
                                sf_pick2 = (input("Which other SF is unrealiable and needs to be removed?:" ))
                                sf_pick2 = int(sf_pick2)
                                k = 0
                            except ValueError:
                                print('Not an interger')
                   
                            
                try:
                 
                    sf_pick = int(sf_pick)

                    if (sf_pick >= 2) and (sf_pick <= maxSF1):

                        fwhmlist[sf_pick-2] = 'che' #does this keep up
                        
              
                        if (answer2 == "yes") and (sf_pick2 >= 2) and (sf_pick2 <= maxSF1):

                            fwhmlist[sf_pick2-2] = 'che' #does this keep up
                        X['fwhmlist'] = fwhmlist

                        X = plot_fwhm(X) #give Hu_0

                        break
                except ValueError:
                    print('Not an interger')
                    True
       # if (isinstance(sf_choose, int)):
            
                
        elif answer == "no":
        
            X = plot_fwhm(X) #give Hu_0
            break #stop functon ? print 'points ok'
        else:
            print("Please enter yes or no.")
    #X['fwhmlist'] = fwhmlist
    
    return(X)
        

def divide_mu0(X):
    mu0 = mu0=4*pi*1e-7
    X['Hc_mu'] = X['Hc']/mu0
    #X['Hu_0_mu'] = X['Hu_0']/mu0
    X['Hu_mu'] = X['Hu']/mu0
    
    return(X)

def FORCinel_colormap(Z):

    #setup initial colormap assuming that negative range does not require extension
    cdict = {'red':     ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                         (0.3193,  102/255, 102/255),
                       (0.563,  204/255, 204/255),
                       (0.6975,  204/255, 204/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255)),

            'green':   ((0.0,  127/255, 127/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  178/255, 178/255),
                        (0.563,  204/255, 204/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  102/255, 102/255),
                       (0.9748,  25/255, 25/255),
                       (1.0, 25/255, 25/255)),

             'blue':   ((0.0,  255/255, 255/255),
                         (0.1387,  255/255, 255/255),
                         (0.1597,  255/255, 255/255),
                         (0.1807,  255/255, 255/255),
                       (0.3193,  102/255, 102/255),
                        (0.563,  76/255, 76/255),
                       (0.6975,  76/255, 76/255),
                       (0.8319,  153/255, 153/255),
                       (0.9748,  76/255, 76/255),
                       (1.0, 76/255, 76/255))}
    
    if np.abs(np.min(Z))<=np.nanmax(Z):#*0.19: #negative extension is not required
        vmin = -np.nanmax(Z)#*0.19
        vmax = np.nanmax(Z)
 
    else: #negative extension is required

        vmin=np.nanmin(Z)
        vmax=np.nanmax(Z)
      
    anchors = np.zeros(10)    
    #anchors[1]=(-0.025*vmax-vmin)/(vmax-vmin)
   # anchors[2]=(-0.005*vmax-vmin)/(vmax-vmin)
    
    anchors[1]=(0.0005*vmax-vmin)/(vmax-vmin)
    anchors[2]=(0.005*vmax-vmin)/(vmax-vmin)   
    anchors[3]=(0.025*vmax-vmin)/(vmax-vmin)
    anchors[4]=(0.19*vmax-vmin)/(vmax-vmin)
    anchors[5]=(0.48*vmax-vmin)/(vmax-vmin)
    anchors[6]=(0.64*vmax-vmin)/(vmax-vmin)
    anchors[7]=(0.80*vmax-vmin)/(vmax-vmin)
    anchors[8]=(0.97*vmax-vmin)/(vmax-vmin)
    anchors[9]=1.0

    
    anchors = abs(anchors)
 

    Rlst = list(cdict['red'])
    Glst = list(cdict['green'])
    Blst = list(cdict['blue'])


    for i in range(9):
        Rlst[i] = tuple((anchors[i],Rlst[i][1],Rlst[i][2]))
        Glst[i] = tuple((anchors[i],Glst[i][1],Glst[i][2]))
        Blst[i] = tuple((anchors[i],Blst[i][1],Blst[i][2]))
        
    cdict['red'] = tuple(Rlst)
    cdict['green'] = tuple(Glst)
    cdict['blue'] = tuple(Blst)
   
    
    cmap = matplotlib.colors.LinearSegmentedColormap('forc_cmap', cdict)
    
    return cmap, vmin, vmax
    
def inter_FORC(X, SF):
    Hu_f = X['Hu_mu'].flatten() #change Hu_0 to Hu
    Hc_f = X['Hc_mu'].flatten()
    Rho_f = X['rho'][SF].flatten() #flatten SF 3 section
    
    #remove nan
    Hu_f = Hu_f[~np.isnan(Hu_f)]
    Hc_f = Hc_f[~np.isnan(Hc_f)]
    Rho_f = Rho_f[~np.isnan(Rho_f)]
    
    step_xi = np.nanmax(X['Hc_mu'])/181.
    step_yi = (np.nanmax(X['Hu_mu']) - np.nanmin(X['Hu_mu']))/146.

    # target grid to interpolate to
    xi = np.arange(0,np.nanmax(X['Hc_mu']),step_xi) #same size needed but not same axes - thinkt and chck
    yi = np.arange(np.nanmin(X['Hu_mu']),np.nanmax(X['Hu_mu']),step_yi) #same size needed but not same axes - thinkt and chck
    xi1,yi1 = np.meshgrid(xi,yi) 


    # interpolate
    # interpolate
    zi = griddata((Hc_f,Hu_f),Rho_f,(xi1,yi1),method='cubic') 

    X['xi1'] = xi1
    X['yi1'] = yi1
    X['zi_{}'.format(SF)] = zi
    return (X)
    
def inter_rho(xi_s_f, yi_s_f, zi_s_f, hys, i): # when call use xi_s etc, i is no. hysteron to test
    xi1_row = xi_s_f[0,:] #should be different
    #print(xi1_row)
    #print(hys[i,0])
    up_hc = xi1_row[xi1_row > hys[i,0]].min()   #test on first data point do with 1D array, value is slightly above
    lo_hc = xi1_row[xi1_row < hys[i,0]].max() #works
    #print(up_hc,lo_hc)
    up_hc_idx = list(xi1_row).index(up_hc) #correct
    lo_hc_idx = list(xi1_row).index(lo_hc)
    #print(up_hc_idx, lo_hc_idx)
    yi1_col = yi_s_f[:,0] #should be different

    up_hi = yi1_col[yi1_col > hys[i,1]].min()   #test on first data point do with 1D array, value is slightly above
    lo_hi = yi1_col[yi1_col < hys[i,1]].max() #works - doesnt work on all dta points

    up_hi_idx = list(yi1_col).index(up_hi) #correct
    lo_hi_idx = list(yi1_col).index(lo_hi)

    x_arr = np.array([xi_s_f[lo_hi_idx,lo_hc_idx], xi_s_f[up_hi_idx, lo_hc_idx], xi_s_f[up_hi_idx, up_hc_idx], xi_s_f[lo_hi_idx, up_hc_idx]])
    y_arr = np.array([yi_s_f[lo_hi_idx,lo_hc_idx], yi_s_f[up_hi_idx, lo_hc_idx], yi_s_f[up_hi_idx, up_hc_idx], yi_s_f[lo_hi_idx, up_hc_idx]])
    z_arr = np.array([zi_s_f[lo_hi_idx,lo_hc_idx], zi_s_f[up_hi_idx, lo_hc_idx], zi_s_f[up_hi_idx, up_hc_idx], zi_s_f[lo_hi_idx, up_hc_idx]])

    #check for nan to remove error
    xarr_sum = np.sum(x_arr)
    xarr_has_nan = np.isnan(xarr_sum)
    yarr_sum = np.sum(y_arr)
    yarr_has_nan = np.isnan(yarr_sum)
    zarr_sum = np.sum(z_arr)
    zarr_has_nan = np.isnan(zarr_sum)    

    
    if (xarr_has_nan != True) and (yarr_has_nan != True) and (zarr_has_nan != True):
        f = interp2d(x_arr, y_arr, z_arr, kind='linear')
        hys[i,3] = f(hys[i,0], hys[i,1]) #swtiched round
    else:
        hys[i,3] = -0.001
    

    #print('hys[i,3]', hys[i,3])
    return hys
    
    
def sym_FORC(X, SF):
    xi1 = X['xi1']
    yi1 = X['yi1']
    zi = X['zi_{}'.format(SF)] # take in SF rho interested in
    yi_axis = np.copy(yi1)

    yi_axis = abs(yi_axis) - 0

    
    indices = np.unravel_index(np.nanargmin(yi_axis),yi_axis.shape)
   

    xi_s = np.copy(xi1)
    yi_s = np.copy(yi1)
    zi_s = np.copy(zi)
  
    x=1
    j=0

    while x < (len(xi1) - indices[0]): #do for rows between max and dist go upto - 177

        j=0
        for j in range(len(xi_s[0])):
            #print(x,j)
            #print(indices[0]-x, indices[0]+x) #(471 - and 471 - x)

            find_mean = np.array([zi_s[indices[0]+x][j],zi_s[indices[0]-x][j]])
            #print('find_mean',find_mean)
            find_mean = find_mean[~np.isnan(find_mean)]
            if (len(find_mean) > 0):
                zi_s[indices[0]+x][j] = np.mean(find_mean)
                #print('zi_s1', zi_s[indices[0]+x][j])
                zi_s[indices[0]-x][j] = zi_s[indices[0]+x][j]

                
            #print('zi_s2', zi_s[indices[0]-x][j])
            j+=1
        x+=1        
    
    lower_x_bound = indices[0] - (len(xi_s) - indices[0])
    upper_x_bound = len(xi_s)
    

    xi_s_cut = xi_s[lower_x_bound:upper_x_bound,:]
    yi_s_cut = yi_s[lower_x_bound:upper_x_bound,:]
    zi_s_cut = zi_s[lower_x_bound:upper_x_bound,:]


    X['xis'] = xi_s_cut
    X['yis'] = yi_s_cut
    X['zis_{}'.format(SF)] = zi_s_cut
    
    
    return(X)
    
#get rid of error outputs in this section 
def sym_norm_forcs(X):

    SFlist = X['SF_list']
    for i in range(len(SFlist)): #loop over each SF 2,3,4,5
        X = inter_FORC(X, SFlist[i])    
        X = sym_FORC(X, SFlist[i])
    for i in range(len(SFlist)):
        X = norm_z2(X, SFlist[i])
    fwhmlist = X['fwhmlist'] #correct FHWM values 
    X['sf_list_correct'] = (X['b']/fwhmlist) 

    return(X)
        

def norm_z2(X, SF):

    z_pre_norm = X['zis_{}'.format(SF)]

    maxval_z = np.nanmax(z_pre_norm)

    minval_z = np.nanmin(z_pre_norm)
    
 
    z_norm = (z_pre_norm)/(maxval_z)

    X['zis_{}'.format(SF)] = z_norm
    return(X)    
    
    
def hys_angles():
    
    angle = random.random()
    #angle = 0.5
    phi = acos(2*angle - 1) #bnot just between 0 and 1.6 becasue times it by 2 and -1. therefore always less than 2, gives negative range too
    if(phi > (pi/2.)): 
        phi = pi-phi
    angle2 = random.random()
    phistatic = acos(2*angle2 - 1)
    if(phistatic > (pi/2.)):
        phistatic = pi-phistatic
    
    angle3 = random.random()
    thetastatic = 2*pi*angle3
    return phi, phistatic, thetastatic
    
def calc_hk_arrays(hys, num, V): #this is called for each inv point, can do at once? num is num_pop
    tm = V['tm']

    hc = np.copy(hys[:,0]) #this could be array hc = hys[:,0]
    tempt = 300.

    hf = ((hc/(sqrt(2)))**(eglip))*(10**(-0.52)) 
    phi = np.copy(hys[:,5]) #this could be an array phi = hys[:,5]
    
    phitemp = (((np.sin(phi))**(2./3.))+((np.cos(phi))**(2./3.)))**(-3./2.) #phi = hys[i,5], phitemp could be array

    gphitemp = (0.86 + (1.14*phitemp)) #gphitemp could be array
  
    hatmp = hc/(sqrt(2)) #this is the hc once its been divided could be array

    ht = hf*(log(tm/tau)) # tau is 10E-9, ht>0 as tm>tau. could be array, tm, tau constants
    
    hktmp = hatmp +ht + (2*hatmp*ht+ht**2)**0.5 #first guess for iteration, its a max guess, can be array
    hktmpmax = hktmp
    
    
    hktmpstore = hktmp
    i=0
    for i in range(int(num)):
        factor = 0.5
        searchme = 1000000.0     
        hstep = (hktmp[i]-hatmp[i])/5.

        while (abs(searchme)> 5):
            searchme = hktmp[i] - hatmp[i] - hktmp[i]*((2*ht[i]*phitemp[i]/hktmp[i])**(1/gphitemp[i])) #
            hktmpstore[i] = hktmp[i]

            if (searchme > 0):
                hktmp[i] = hktmp[i] - hstep
            else:
                hktmp[i] = hktmp[i] + hstep
                hstep = hstep*factor #this else should be for the Hktmp = hktmp - hstep

    hkphi = hktmpstore #think function before made arrys so just do array opertatiosn now
    hk = hkphi[:int(num)]/phitemp[:int(num)]

    hys[:int(num),9] = hk #unsure if this will it fill correctly

    return(hys)
    
    
def pop_hys(num_hys, X, V, SF): #this function will populate all from beggining each time
    #populate hysterons
    
    corr = X['sf_list_correct'][SF - 2]
    hys = np.zeros((num_hys,11)) 
    
    #Hc, Hi, random no (Rho test), rho_frominter_test, mag (1), angle, static angle, mag times angle, Hf, vol
    num_pop = num_hys/2
    
    xi_s_cut = X['xis']
    yi_s_cut = X['yis']
    zi_s_cut = X['zis_{}'.format(SF)] 
    maxHc = np.nanmax(xi_s_cut) #change to hu values, change to cut limits and sometmes slightly out and stopped 
    maxHi = np.nanmax(yi_s_cut)

    xlim_res = X['reset_limit_hc']/1000.
    ylim_res = X['reset_limit_hi']/1000.
    maxHc = min(xlim_res,X['Hc2'])
    maxHi = min(ylim_res,X['Hb2'])

   
    i=0

    while (i < int(num_pop)):
        z1 = random.random()
        z2 = random.random()
        z3 = random.random()
  
        hys[i,0] = (z2*maxHc)/mu0

        hys[i,1] = (z3*maxHi)/mu0
        
        
        hys = inter_rho(xi_s_cut, yi_s_cut, zi_s_cut, hys, i) #ztestnorm
        hys[i,1] = hys[i,1]*corr
        hys[i,5], hys[i,6], hys[i,7] = hys_angles() #calc for half hys
        
        if ((hys[i,1]) <= (hys[i,0])) and (hys[i,3] >= 0) and (hys[i,3] <= 1): #add in if hys[i,3] is positive - may not need
            i +=1 #cna have if statements below which can sto pthis #this line to change - remove hys[i,2] <= hys[i,3]

    hys = calc_hk_arrays(hys, int(num_pop), V) # try to calc hk using arrrys 

    hys[:,4] = 1
 
    hys[:,8] = hys[:,5]*hys[:,4] #move to be more efficent
    num_pop = int(num_pop)
    j=0
    for j in range(num_pop):
        hys[(j+num_pop),:] = hys[j,:]
        hys[j+num_pop,1] = -hys[j,1]
        j+=1


    return hys, num_pop

     
    
@jit(nopython = True, parallel=True)
def block_loc(var_1, hys, blockg, boltz):

    #unpack var_1
    num_hyss = int(var_1[0])
    beta = float(var_1[1])
    blockper = float(var_1[2])
    temp = float(var_1[3])
    aconst = float(var_1[4])
    curie_t = float(var_1[5])
    rate = float(var_1[6])    
    tempmin = float(var_1[7])
    field = float(var_1[8])
    tm = float(var_1[9])
    heating = float(var_1[10])

    tau = 10e-9
    roottwohffield = 2**(0.5)
    hfstore = np.zeros((num_hyss))
    histore = np.zeros((num_hyss))    
    hcstore = np.zeros((num_hyss))   
    blocktemp = np.zeros((num_hyss))
    i=0
 
    for i in range(num_hyss): #dynamic contributions - to hc dormann 1988 used?

        hc=(sqrt(2))*(hys[i,9])*beta #test and time HCbysqrt2 to get hc and matfhes similar order mag to hys[i,0/mu0 -> hc in same units using in hk - seems correct]
       
        hcstore[i] = hc/(sqrt(2)) #different to hys[0]

        hi = hys[i,1]*beta*blockper #divide here by mu0

        histore[i] = hi/(sqrt(2)) #should all be zero, without blockper, all look milar to hys[1,0] in mg
 
        phitemp=((sin(hys[i,5])**(2./3.))+(cos(hys[i,5])**(2./3.)))**(-3./2.) #phitemp from hys[5] -> phi

        g=0.86+1.14*phitemp

        hf=((hys[i,9]**eglip))*(10**(-0.52))*temp/(300*beta) #where this eq come from
   
        hfstore[i] = hf #values at expected as hc/mu0

        if (rate == 1): #rate first set to 1 so shold do this

            r = (1./aconst)*(temp-tempmin)
  
            tm = (temp/r)*(1 - (temp/(curie_t+273)))/log((2*temp)/(tau*r)*((1. - (temp/(curie_t+273)))))      

            
            if (tm == 0.0): #unsure
                tm = 60.0

        ht = (roottwohffield)*hf*(log(tm/tau)) #new tm 

        bracket = 1-(2*ht*phitemp/hc)**(1/g)
        
        hiflipplus = hc*bracket+field*(roottwohffield) # using hs=-hi then field is +ve not -ve, 

        hiflipminus=-hc*bracket+field*(roottwohffield) #see which way fields flips
   
        if (hc >= (2*ht*phitemp)): #still loop over each hys, +1

            if ((hi > hiflipminus) and (hi < hiflipplus)): #+2 blocked
              

                if ((blockg[i] == 0) or (blockg[i] == 2) or (blockg[i] == -2)): #+3 prev blocked until this point
                    
                    if (hi >= (field*roottwohffield)): #+4
                        
                        blocktemp[i] = -1
                       
                    else:

                        blocktemp[i] = 1 #end +3 unsure if sholud ended both or just one
                      
                elif (blockg[i] == -1): #this line, already block stay , not need
                 
                    blocktemp[i] = -1
            

                elif (blockg[i] == 1):
  
                    blocktemp[i] = 1
                   
                else:
                    #write to screen, blockg[i]
                    print(blockg[i], blocktemp[i]) #see if work - not need
                    print('----', i)
                #end if +2, actually back to +2 if statement, still if hi
            elif (hi >= hiflipplus):#else field blocking above ht, this is hi test hiflip etc
               
                blocktemp[i] = -2
              
            else:
               
                blocktemp[i] = 2
        else: #hc < 2*ht*phitemp. this is correctm meaning else above isnt

            if ((hi < hiflipminus) and (hi > hiflipplus)): 
               
                blocktemp[i] = 0
                if (heating == 1):
                
                    boltz[i] = 0.0

            else: #field blocking - below hc
                if (hi >= hiflipminus):

                    blocktemp[i] = -2
                else:

                    blocktemp[i] = 2

    return hfstore, histore, boltz, blocktemp  


@jit(nopython = True, parallel=True)
def block_val(hys, histore, hfstore, blocktemp, beta, num_hyss, boltz, blockg, field):
   
    i=0
    totalm = 0.0
    blockper = 0   
    for i in range(num_hyss):
        x = blocktemp[i]
        blockg[i] = x
        absblockg = abs(blockg[i])
        
        if (absblockg == 1): #if blockg 1 or -1
            if (boltz[i] < 0.00000001) and (boltz[i] > -0.000000001): #only zero

                boltz[i] = tanh((field - histore[i])/hfstore[i])
  
        if (blockg[i] == -2):
         
            moment = -0 #changed from zero to minues zero
        elif (blockg[i] == 2):

            moment = 0
        else:

            moment = blockg[i] #where does moment come from? - set to blockg if blockg not equal to 1

        #bit before is TRM acqution  

        totalm = totalm + abs(moment)*abs(cos(hys[i,5]))*hys[i,3]*beta*(boltz[i]) #add in forcdist (hys[i,3])


        blockper=blockper+abs(moment)*hys[i,3]*1.0 
        i+=1

        
    blockper=blockper/(1.0*np.sum(hys[:,3]))

    return blockper, totalm, boltz, blockg
    

def blockfind(temp, field, afswitch, V, X): #try removing counttime as input/output

    hys = V['hys']
    num_hyss = V['num_hyss']
    hcstore = V['hcstore']
    histore = V['histore']
    beta = V['beta']
    rate = V['rate']
    aconst = V['aconst']
    tempmin = V['tempmin']
    
    heating = V['heating']
    
    tm = V['tm'] #try setting this for use in AF demag

    hfstore = np.zeros(num_hyss)

    blockper = V['blockper'] 

    blocktemp = V['blocktemp'] 
    boltz = V['boltz']
    blockg = V['blockg']

    curie_t = V['curie_t']

    totalm = V['totalm']


    #call first blockfind function
    var_1 = np.array((num_hyss, beta, blockper, temp, aconst, curie_t, rate, tempmin, field, tm, heating))

    hfstore, histore, boltz, blocktemp = block_loc(var_1, hys, blockg, boltz)  #inside here loops over each hystern   
    
    blockper, totalm, boltz, blockg = block_val(hys, histore, hfstore, blocktemp, beta, num_hyss, boltz, blockg, field)
    
    V['blockper'] = blockper
    V['blocktemp'] = blocktemp
    V['boltz'] = boltz
    V['blockg'] = blockg
    V['totalm'] = totalm
    V['tm'] = tm

    return(V) #totalm
    #end of blocktemp function - inputs etc - global        
  
def ms_t_file():
    ms_t = FileChooser()
    display(ms_t)

    return(ms_t)
 
def read_file2(ms_t, V):
    curve = True
    try: 
        ms_t_file = ms_t.selected
        temp_ms = []
        lista_ms = []

        with open(ms_t_file) as my_file:
           # my_file.readline()
            for line in my_file:
                line_s = line.split()
                temp_ms.append(float(line_s[0]))
                lista_ms.append(float(line_s[1]))


        
        temp_ms = np.array(temp_ms)
        list_ms = np.array(lista_ms)
        
    except:
        curve = False
        
    if (curve == True):
        list_ms = np.where(list_ms < 0.0, 0.0, list_ms)
        V['temp_ms'] = temp_ms + 273
        V['list_ms'] = list_ms
        idx_rt = (np.abs(temp_ms - 27)).argmin()
        ms_R = list_ms[idx_rt]
        V['ms_R'] = ms_R 
        
        
        #simple case to find curie T
        i=0
        for i in range(len(list_ms)):
            if (list_ms[i] == 0):
                curie_idx = i-1
                break
            i+=1
        curie_t = temp_ms[curie_idx] 
        
        V['curie_t'] = curie_t
   # my_file.close()

    return(V, curve)


def plots_correction_factor(V, X):
    
    track2 = V['track2']
    track = V['track']

    #find inverse 
    find_max = np.copy(np.argmin(V['temp_h'][0][:track2]))
    use_max_lab = np.copy(V['heat'][0][:track2])
    use_max_nat = np.copy(V['heat'][1][:track2])
    rev_mag_lab = use_max_lab[find_max] - (V['heat'][0][:track2])
    rev_mag_nat = use_max_nat[find_max] - (V['heat'][1][:track2])    
    #m_ratio = np.zeros_like(rev_mag_lab)
    #function to calcualte the difference- same temp steps
    #make m_ratio exact same 
   
    m_ratio = np.zeros_like(rev_mag_lab)
    rev = 0
    for rev in range(len(rev_mag_lab)):
        if (rev_mag_lab[rev] > 1E-10):
            m_ratio[rev] = rev_mag_nat[rev]/rev_mag_lab[rev]    
        else:
            m_ratio[rev] = 1.
            

    
    #truncate m_ratio to the length of track etc 
    #running average of ratio
    
    #quality control on m_ratio
    
    #print(m_ratio)
    #print(np.nanmax(rev_mag_nat), np.nanmax(rev_mag_lab))
    #print(0.01*(np.nanmax(rev_mag_nat) - np.nanmax(rev_mag_lab)))

    for i in range(len(m_ratio)):
        if (rev_mag_nat[i] < (0.002*(np.nanmax(rev_mag_nat)))) and  (rev_mag_lab[i] < (0.002*np.nanmax(rev_mag_lab))): #((rev_mag_nat[i] - rev_mag_lab[i]) < 0.04*np.max(rev_mag_lab)):
            m_ratio[i] = 1
    
    #print(m_ratio)
  
    m_ratio_av = np.zeros_like(m_ratio) 

    for i in range(len(m_ratio)):
        #sum surr bits
        #divide by number
        if (i < 3):
            lb = 0
        else:
            lb = i - 3
        if (i> len(m_ratio) -3): # test these exact to get correct number or just know length of track etc and use limits 
            ub = len(m_ratio)
        else:
            ub = i+3
        #print(m_ratio[i], m_ratio[lb:ub], np.mean(m_ratio[lb:ub]))
        m_ratio_av[i] = np.mean(m_ratio[lb:ub+1]) # check if need to add one on etc test correct number either side

    #calculated runing aveage - put in dictionary and have as function 
   
    V['m_ratio_av'] = m_ratio_av
    return(V)
    

def fix_files(path, V):
    track2 = V['track2']
    temp_c = V['temp_h'][0][:track2] - 273
    m_ratio_av = V['m_ratio_av']
    files_in = []
    for root, dir, file in os.walk(path):
        for f in file:
            #print(f)
            if re.match('.*\.tdt', f):
           #     print(f)
                files_in.append(path+'/'+f)
                #print(root f "/" file) #run_existing_script1 root + "/" file

                #run_existing_script2 root + "/" file
        #files_in[i].close()
    #open new file

    for i in range(len(files_in)):
        name_in = []
        intensity_in = []
        temp_in = []
        dec_in = []
        inc_in = []
        with open(files_in[i]) as my_file:
            count = 0
            for line in my_file:
               # print(line)
                if (count == 0):
                    header = line.split()
                    top_line = header
                  #  print(line, 'header1')
                elif (count == 1):
                    header2 = line.split()
                 #   print(header2)
                    col1 = header2[0]
                    col2 = header2[1]
                    col3 = header2[2]
                    col4 = header2[3] 
                    col5 = header2[4] 
                  #  print(line, 'header2')
                else:
                   # print(count, 'main')
                   # print(line_v)
                    line_v = line.split()
                    #print('1 =', line_v[0], '2=', line_v[1],'3=', line_v[2], '4=', line_v[3],'5=', line_v[4])
                    name_in.append(line_v[0])

                    temp_in.append(float(line_v[1]))
                    intensity_in.append(float(line_v[2]))
                    dec_in.append(float(line_v[3]))
                    inc_in.append(float(line_v[4]))     
                count+=1

            intensity_in_a = np.array(intensity_in)
            intensity_out = np.copy(intensity_in_a)
            temp_in_a = np.array(temp_in)
            dec_in_a = np.array(dec_in)
            inc_in_a = np.array(inc_in)

            xt=np.cos(dec_in_a*pi/180.0)*np.cos(inc_in_a*pi/180.0)
            yt=np.sin(dec_in_a*pi/180.0)*np.cos(inc_in_a*pi/180.0)
            zt=np.sin(inc_in_a*pi/180.0)
           # print(type(xt))

            #edit the data file using what I have


            j=0
            for j in range(len(temp_in_a)):
                if (str(temp_in_a[j]).endswith('.1')) or (str(temp_in_a[j]).endswith('.2')):
            #loop over again and find .1 less than this 

                    temp_in_field = temp_in_a[j]

                    x = j
                    while (str(temp_in_a[x]).endswith('.0') == False):
                    
                        loc_out = x
                        x-= 1
                    loc_out = x
                 
                    temp_out_field = float(temp_in_a[loc_out])
                    #collec x,y,z etc mag at each step

                   # print(temp_in_field)
                    cf_loc1 = (np.abs((temp_c - temp_in_field))).argmin() #closest temp from my acquisition curves - sae indicies as array
                    if ((temp_c[cf_loc1]) > temp_in_field):
                        cf_loc2 = cf_loc1-1
                    else:
                        cf_loc2 = cf_loc1+1
                    if (cf_loc1 < len(m_ratio_av)) and (cf_loc2 < len(m_ratio_av)):
                        cf = np.interp(temp_in_field, [temp_c[cf_loc1], temp_c[cf_loc2]], [m_ratio_av[cf_loc1], m_ratio_av[cf_loc2]])
                    else:
                        cf = m_ratio_av[cf_loc1]
                    #cf = m_ratio_av[cf_loc]
                    #print(temp_c[cf_loc])
                    in_field = intensity_in_a[j]

                    
                    out_field = intensity_out[loc_out]
                    intensity_out[j] = sqrt(((cf*xt[j]*in_field + xt[loc_out]*out_field*(1-cf))**2) + ((cf*yt[j]*in_field+yt[loc_out]*out_field*(1-cf))**2) + ((cf*zt[j]*in_field+zt[loc_out]*out_field*(1-cf))**2))
                j+=1
                   # print(cf)
            #newint wll update .1 step

            file_out_name  = files_in[i][:(len(files_in[i])-4)]
            f = open('{}_U.tdt'.format(file_out_name), 'w')
            f.write('Thellier-tdt' + "\n")
            f.write(str(col1) + "\t" + str(col2) + "\t" + str(col3) + "\t" + str(col4) + "\t" + str(col5) + '\n') 
            k=0
            for k in range(len(inc_in)):
                f.write(name_in[k] + "\t" + str("{:.1f}".format(temp_in[k])) +  "\t" + str("{:.12f}".format(intensity_out[k])) + "\t" + str(dec_in[k]) + "\t" + str(inc_in[k]) + "\n")
            f.close()
    return
    
    
def TRM_acq(X, V, curve, path):
   #just once turn temp_ms into K 
    
  
    SF = X['SF']
    
    if (curve == True):
        temp_ms = V['temp_ms']
        list_ms = V['list_ms']
        curie_t = V['curie_t']
    else:
        curie_t = V['curie_t']
          
    V['heating'] = 0.
    mu0 = 4*pi*1e-7
    #ms0 = 491880.5
    kb = 1.3806503e-23
    tau = 10e-9
    roottwohffield = 2**(0.5)  

    eglip=0.54
    eglid=+0.52+(log10(hf)-log10(10000.0)*eglip)

    afone = 1
    afzero = 0

    blockper = 0.0
    V['blockper'] = blockper

    num_hyss = 300000
    V['num_hyss'] = num_hyss

    tempmax = float(V['curie_t'] + 273)
    tempmin=300 #lines 116
    V['tempmin'] = tempmin
    V['tempmax'] = tempmax
    tempstep=1

    hcstore = np.zeros(num_hyss)
    V['hcstore'] = hcstore
    histore = np.zeros(num_hyss)
    V['histore'] = histore

    field = (50E-6)/mu0
    ifield = 0
    fields = np.zeros((100))


    trm_a = np.zeros((2, 1000))
    V['trm_a'] = trm_a
    len_a_trm = np.zeros((10))
    V['len_a_trm'] = len_a_trm
    temp_a = np.zeros((2, 1000))
    V['temp_a'] = temp_a
    heat = np.zeros((2, 1000))
    V['heat'] = heat
    temp_h = np.zeros((2, 1000))
    V['temp_h'] = temp_h    
    
    
    V['variable_change'] = np.zeros((2))
    V['variable_change'][0] = X['lab_cool']
    V['variable_change'][1] = X['nat_cool']
    

    if (curve == True):
        list_ms_2 = np.copy(list_ms)

        #temp closest to zero 
        mso_loc1 = (np.abs((temp_ms - 300.0))).argmin()

        if (mso_loc1 != 0):
            if ((temp_ms[mso_loc1]) > 300.0) and (temp_ms[mso_loc1-1] < 300.0): #if temp ms is greater tha zero and the second smalled is too 
                mso_loc2 = mso_loc1-1
            else:
                mso_loc2 = mso_loc1+1
            ms0 = np.interp(300.0, [temp_ms[mso_loc1], temp_ms[mso_loc2]], [list_ms[mso_loc1], list_ms[mso_loc2]])  
        elif (temp_ms[mso_loc1] > 300.0) and (temp_ms[mso_loc1 +1 ] > 300.0):
            ms0 = list_ms[mso_loc1]
        elif (temp_ms[mso_loc1] < 300.0) and (temp_ms[mso_loc1 +1 ] < 300.0):
            ms0 = list_ms[mso_loc1]

        else:
            mso_loc2 = mso_loc1 +1
            ms0 = np.interp(300.0, [temp_ms[mso_loc1], temp_ms[mso_loc2]], [list_ms[mso_loc1], list_ms[mso_loc2]])  

        V['msR'] = ms0 #this is really ms_R
        beta_a = list_ms/(ms0) #edit to be from 300K
        V['beta_a'] = beta_a    
    

    blockg = np.zeros(num_hyss) #line 349
    boltz = np.zeros(num_hyss)
    blocktemp = np.zeros(num_hyss) 

    V['blocktemp'] = blocktemp
    V['boltz'] = boltz
    V['blockg'] = blockg
    V['totalm'] = 0


    i=0 #is this i=1 or i=0?
    temp=300
    tempt=300

    tm = 0.2 #change from 0.2 to 0.1 for line 43 # typical forc measuring time, poss use 0.1
    V['tm'] = tm


    hys1, num_pop = pop_hys(num_hyss, X, V, SF) #called later here?       

    ct = 0
    for ct in range(len(V['variable_change'])):
        field = (50E-6)/mu0
        ifield = ct
        ac = V['variable_change'][ct]

        blockg = np.zeros(num_hyss) #line 349
        boltz = np.zeros(num_hyss)
        blocktemp = np.zeros(num_hyss) 

        V['blocktemp'] = blocktemp
        V['boltz'] = boltz
        V['blockg'] = blockg
        V['totalm'] = 0


        i=0 
        temp=300
        tempt=300

        tm = 0.2 #change from 0.2 to 0.1 for line 43 # typical forc measuring time, poss use 0.1
        V['tm'] = tm

        temp = tempmax
        rate = 1 
        V['rate'] = rate


        afswitch = afzero

        V['hys'] = copy.deepcopy(hys1)
        #TRM acquition
        tm = 0

        V['tm'] = tm #to use for af demag bit delcare here
        track = 0
        temp = curie_t + 273 - 1

        while (temp > tempmin):
            aconst=(-ac*60.0*60.0)/(log(0.01*(tempmin)/(tempmax-tempmin))) #line 708 
            V['aconst'] = aconst

            if (curve == True):
                beta_loc1 = (np.abs((temp_ms - temp))).argmin()
               # print('betaloc1', beta_loc1)
                if (beta_loc1 != 0):
                    if ((temp_ms[beta_loc1]) > temp) and (temp_ms[beta_loc1-1] < temp): #if temp ms is greater tha zero and the second smalled is too 
                        beta_loc2 = beta_loc1-1
                        beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
                    elif (beta_loc1 != (len(temp_ms) -1)):
                        beta_loc2 = beta_loc1+1
                        beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])   
                    elif (beta_loc1 == (len(temp_ms)-1)):
                        beta_u = beta_a[beta_loc1] # if at max length just take as value 
                elif (temp_ms[beta_loc1] > temp) and (temp_ms[beta_loc1 +1 ] > temp):
                    beta_u = beta_a[beta_loc1]
                elif (temp_ms[beta_loc1] < temp) and (temp_ms[beta_loc1 +1 ] < temp):
                    beta_u = beta_a[beta_loc1]
                else:
                    beta_loc2 = beta_loc1 +1
                    beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
                beta = beta_u
             
            else:
                beta = (1-(temp-273)/curie_t)**0.43


            V['beta'] = beta

            V = blockfind(temp, field, afzero, V, X) #does counttime change in blockfind

            V['trm_a'][ifield][track] = V['totalm']
            V['temp_a'][ifield][track] = temp

            track +=1
           

            tempstep1 = tempstep*2
            temp = temp - tempstep1
 
        rate = 0
        V['rate'] = rate
        tm = 60
        V['tm'] = tm
        temp = tempmin + 0.1 #tempstep #print these last temps and check correct 
   
        if (curve == True):
            beta_loc1 = (np.abs((temp_ms - temp))).argmin()
            if (beta_loc1 != 0):
                if ((temp_ms[beta_loc1]) > temp) and (temp_ms[beta_loc1-1] < temp): #if temp ms is greater tha zero and the second smalled is too 
                    beta_loc2 = beta_loc1-1
                else:
                    beta_loc2 = beta_loc1+1
                beta_u = np.interp(tempt, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
            elif (temp_ms[beta_loc1] > temp) and (temp_ms[beta_loc1 +1 ] > temp):
                beta_u = beta_a[beta_loc1]
            elif (temp_ms[beta_loc1] < temp) and (temp_ms[beta_loc1 +1 ] < temp):
                beta_u = beta_a[beta_loc1]
            else:
                beta_loc2 = beta_loc1 +1
                beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
            beta = beta_u
        else:

            beta = (1-(temp-273)/578.0)**0.43

        V['beta'] = beta
      #  beta = (1 - (temp - 273)/578.0)**0.43
        fieldzero = 0.0
        V = blockfind(temp, fieldzero, afzero, V, X) #check if totalm, spcounti work still
        
        V['trm_a'][ifield][track] = V['totalm'] #add on zero case for end of TRM 
        V['temp_a'][ifield][track] = temp
        track+=1


        #now acquired TRM need to heat it up for each temp heat to temp 0 field and re-measure again 0 field at room temperature loop over heating and save V['heat_a'] in same way as trm_acq and temp_h in same way as temp_a 
        track2 = 0
        temp = 300.1
        
        rate = 1.
        V['rate'] = rate 
        
        ac = float(V['variable_change'][0]) #X['lab_cool']
        V['heating'] = 1.

        while (temp < (curie_t + 273)):
            aconst=(-ac*60.0*60.0)/(log(0.01*(tempmin)/(tempmax-tempmin))) #line 708 
            V['aconst'] = aconst
          #  print('aconst', V['aconst'])
            if (curve == True):
                msR = V['ms_R']
                beta_loc1 = (np.abs((temp_ms - temp))).argmin()
               # print('betaloc1', beta_loc1)
                if (beta_loc1 != 0):
                    if ((temp_ms[beta_loc1]) > temp) and (temp_ms[beta_loc1-1] < temp): #if temp ms is greater tha zero and the second smalled is too 
                        beta_loc2 = beta_loc1-1
                        beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
                    elif (beta_loc1 != (len(temp_ms) -1)):
                        beta_loc2 = beta_loc1+1
                        beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])   
                    elif (beta_loc1 == (len(temp_ms)-1)):
                        beta_u = beta_a[beta_loc1] # if at max length just take as value 
                elif (temp_ms[beta_loc1] > temp) and (temp_ms[beta_loc1 +1 ] > temp):
                    beta_u = beta_a[beta_loc1]
                elif (temp_ms[beta_loc1] < temp) and (temp_ms[beta_loc1 +1 ] < temp):
                    beta_u = beta_a[beta_loc1]
                else:
                    beta_loc2 = beta_loc1 +1
                    beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
                beta = beta_u
             
            else:
                beta = (1-(temp-273)/curie_t)**0.43

      
            V['beta'] = beta
            field = 0.0
            #print('temp', temp)
            V['heating'] = 1.
            V = blockfind(temp, field, afzero, V, X) #does counttime change in blockfind
            #blockfind at high temperature 
            #now do at 300 K 
            if (curve == True):
                beta_loc1 = (np.abs((temp_ms - 300.1))).argmin()
                if (beta_loc1 != 0):
                    if ((temp_ms[beta_loc1]) > 300.1) and (temp_ms[beta_loc1-1] < 300.1): #if temp ms is greater tha zero and the second smalled is too 
                        beta_loc2 = beta_loc1-1
                    else:
                        beta_loc2 = beta_loc1+1
                    beta_u = np.interp(300.1, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
                elif (temp_ms[beta_loc1] > 300.1) and (temp_ms[beta_loc1 +1 ] > 300.1):
                    beta_u = beta_a[beta_loc1]
                elif (temp_ms[beta_loc1] < 300.1) and (temp_ms[beta_loc1 +1 ] < 300.1):
                    beta_u = beta_a[beta_loc1]
                else:
                    beta_loc2 = beta_loc1 +1
                    beta_u = np.interp(300.1, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
                beta = beta_u
            else:

                beta = (1-(300.1-273)/578.0)**0.43
            V['beta'] = beta          
            V['heating'] = 0.0
            V = blockfind(300.1, field, afzero, V, X) #does counttime change in blockfind       
           # print(V['totalm'])
            V['heat'][ifield][track2] = V['totalm']
            V['temp_h'][ifield][track2] = temp

            track2 +=1
          #  print('temp_h', temp, 'mag', V['totalm']

          
            tempstep1 = tempstep*5
            temp = temp + tempstep1

        #high temp 
        
        temp = tempmin
        

        j = ifield 
    

        V['heating'] = 0.
        ifield = ifield +1 #end of loop for field
        V['aconst'] = aconst

        V['ifield'] = ifield
        V['fields'] = fields
        V['track'] = track
        V['track2'] = track2        
       
        ct +=1
        
        
    #do all at once so can be forgoteen 

        
        
    return    
    
    
    