from numba import jit
import numpy as np
from scipy.linalg import lstsq
from matplotlib.figure import figaspect
from math import acos, pi, degrees, sin, cos, sqrt, log, log10, tanh, remainder
from scipy.interpolate import interp2d
import random
import codecs as cd
from zipfile import ZipFile
import shutil

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


mu0 = 4*pi*1e-7
kb = 1.3806503e-23
tau = 10e-9
roottwohffield = 2**(0.5)  

affield = 0
affmax = 0.0
af_step_max = 0.0
flatsub=0
flat=0


afone = 1
afzero = 0


#find and upzip zipped file 
def open_zip_file():
    path = os.getcwd()
    try:
        os.mkdir('raw_data')
    except:
        print('This step has already been carried out and a raw_data folder exists. To add more files, restart the kernel and re-run the previous cells too.')
    for item in os.listdir(path): # loop through items in dir
        
        if item.endswith('.zip'): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            zip_ref = ZipFile(file_name) # create zipfile object
            zip_ref.extractall(os.path.join(path+ os.sep,'raw_data')) # extract file to dir
            zip_ref.close() # close file
            #os.remove(file_name) # delete zipped file
    return

def find_FORC_MsT(X, V):
    path = os.getcwd()
    path = os.path.join(path+os.sep,'raw_data')
    forc_file = []
    for root, dir, file in os.walk(path):
        i =0
        for f in file:
            if re.match('.*.frc', f):
                forc_file.append(os.path.join(root + os.sep,f))
                i+=1
    X['fn'] = forc_file[0]
    print('Location of the FORC file: ', X['fn'])  

    mst_file = []
    for root, dir, file in os.walk(path):
        i =0
        for f in file:
            if re.match('.*.dat', f):
                mst_file.append(os.path.join(root + os.sep,f))
                i+=1

    if (len(mst_file) != 0):
        curve = True # have Ms-T curve 
        X['ms_file'] = mst_file[0]
        print('Location of the Ms-T data file: ', X['ms_file'])


        temp_ms = []
        lista_ms = []

        with open(X['ms_file']) as my_file:
 
            for line in my_file:
                line_s = line.split()
                temp_ms.append(float(line_s[0]))
                lista_ms.append(float(line_s[1]))

        temp_ms = np.array(temp_ms)
        list_ms = np.array(lista_ms)
    
        list_ms = np.where(list_ms < 0.0, 0.0, list_ms)
        V['temp_ms'] = temp_ms + 273
        V['list_ms'] = list_ms
        idx_rt = (np.abs(temp_ms - 27)).argmin()
        ms_R = list_ms[idx_rt]
        V['ms_R'] = ms_R 
        
        i=0
        for i in range(len(list_ms)):
            if (list_ms[i] == 0):
                curie_idx = i-1
                break
            i+=1
        curie_t = temp_ms[curie_idx] 
        
        V['curie_t'] = curie_t
    else:
        print('No Ms-T curve data file could be found.')
        curve = False


    tdt_file = []
    print('Thellier files (.tdt) found:')
    i =0
    for root, dir, file in os.walk(path):        
        for f in file:
            if re.match('.*.tdt', f):
                tdt_file.append(os.path.join(root + os.sep,f))
                print(f)
                i+=1
    if (i == 0):
        print('No Thellier files (.tdt) have been found.')

    return X, V, curve   

def user_input(X, V, curve):
    lab_t = 0
    while (lab_t == 0):
        try:
            lab_cool_time = (input("Input the lab cooling time in hours:" )) 
            lab_cool_time = float(lab_cool_time) #ask for interger and check it is an interger, if not ask again
            lab_t = 1
        except ValueError:
            print('Not a number. Please input an number.')
            lab_t = 0

    nat_t = 0
    while (nat_t == 0):
        try:
            nature_cool_time = (input("Input the natural cooling time in hours:" )) 
            nature_cool_time = float(nature_cool_time) #ask for interger and check it is an interger, if not ask again
            nat_t = 1
        except ValueError:
            print('Not a number. Please input an number.')
            nat_t = 0
    X['lab_cool'] = lab_cool_time
    X['nat_cool'] = nature_cool_time

    #FORC limits
    hc_l = 0
    while (hc_l == 0):
        try:
            max_input_hc = (input("Set maximum hc (mT) using the above FORC diagram:" ))
            max_input_hc = float(max_input_hc) #ask for interger and check it is an interger, if not ask again
            hc_l = 1
        except ValueError:
            print('Not a number. Please input an number.')
            hc_l = 0

    hi_l = 0
    while (hi_l == 0):
        try:
            max_input_hi = (input("Set absolute maximum hi (mT) using the above FORC diagram:" ))
            max_input_hi = float(max_input_hi) #ask for interger and check it is an interger, if not ask again
            hi_l = 1
        except ValueError:
            print('Not a number. Please input an number.')
            hi_l = 0
    X['reset_limit_hc'] = float(max_input_hc)
    X['reset_limit_hi'] = float(max_input_hi)


    #if no Ms-T curve use this 
    if (curve == False):
        c_l = 0
        while (c_l == 0):
            try:
                curie_t = (input("Input Curie temperature in \xb0C:" ))
                curie_t = float(curie_t) #ask for interger and check it is an interger, if not ask again
                c_l = 1
            except ValueError:
                print('Not a number. Please input an number.')
                c_l = 0
        V['curie_t'] = curie_t #if no Ms-T curve - set Curie T here        

    return X,V



#FORC file read 

def FORCfile():
    fc = FileChooser()
    display(fc)
    return fc

def proccess_all(X):

    #4
    #process data
    style = {'description_width': 'initial'}
    fn = X['fn'] 
    sample, unit, mass = sample_details(fn)

    sample_widge = widgets.Text(value=sample,description='Sample name:',style=style) 
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
    X["slope"] = slope 

    if X['unit']=='Cgs': 
        X = CGS2SI(X)
    
    X = drift_correction(X) 
   
    X = slope_correction(X)
    X = remove_fpa(X) 
 
    X = remove_lpa(X)
    X = lowerbranch_subtract(X)

    return(X)
    
def prod_FORCs(X): #process and name forc from file 
    name = X['fn'].split("/")
    name2 = name[-1].split('.')
    X['name'] = name2[0]


    maxSF = 5
    X['maxSF1'] = maxSF
   
    X = create_arrays(X, maxSF)
    X = nan_values2(X)
   
    for sf in range(2, maxSF+1):
        X = calc_rho(X, sf)
        sf+=1   
    X = nan_values(X, maxSF) 
    X = rotate_FORC(X)
    return(X)    
    



def parse_header(file,string):

    output=-1 
    with cd.open(file,"r",encoding='latin9') as fp:
        for line in lines_that_start_with(string, fp): 
            idx = line.find('=') 
            if idx>-1.: 
                output=float(line[idx+1:]) 
            else: 
                idx = len(string) 
                output=float(line[idx+1:])  

    return output


def parse_measurements(file):


    dum=-9999.99 
    N0=int(1E6) 
    H0=np.zeros(N0)*np.nan #initialize NaN array to contain field values
    M0=np.zeros(N0)*np.nan #initialize NaN array to contain magnetization values
    H0[0]=dum #first field entry is dummy value
    M0[0]=dum #first magnetization entry is dummy value 

    count=0 #counter to place values in arrays
    with cd.open(file,"r",encoding='latin9') as fp: 
        for line in find_data_lines(fp): 
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

    return [line for line in fp if ((line.startswith('+')) or (line.startswith('-')) or (line.strip()=='') or line.find(',')>-1.)]
def lines_that_start_with(string, fp):
 
    return [line for line in fp if line.startswith(string)]


#### PREPROCESSING OPTIONS ####
def options(X):
    style = {'description_width': 'initial'} #general style settings

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

    for i in range(no_FORC): #find main points
        for j in range(max_FORC_len): #find each j indice
            #locate smoothing grids

            cnt = 0
            h1 = min(i, SF) #row from SF below and SF above
            h2 = min(SF, (no_FORC - i)) #loop over all points
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
                              
                            A[cnt, 0] = 1.
                            A[cnt, 1] = Hr_A[i+h][j+k+h] - Hr_A[i][j]
                            A[cnt, 2] = (Hr_A[i+h][j+k+h] - Hr_A[i][j])**2.
                            A[cnt, 3] = H_A[i+h][j+k+h] - H_A[i][j]
                            A[cnt, 4] = (H_A[i+h][j+k+h] - H_A[i][j])**2.
                            A[cnt, 5] = (Hr_A[i+h][j+k+h] - Hr_A[i][j])*(H_A[i+h][j+k+h] - H_A[i][j])
                            b[cnt] = M_A[i+h][j+k+h]

                            cnt+=1 #count number values looped over
                A = A[~np.isnan(A).any(axis=1)]
                b = b[~np.isnan(b)]
                if (len(A)>=2): #min no. points to need to smooth over
 
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

    xp = x*1000
    yp = y*1000

    con = np.linspace(0.1, 1, 9)



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


    plt.title('{} FORC diagram'.format(sample_name))

    plt.tight_layout()
    plt.show

    return     
    
def plot_sample_FORC2(x, y, z, SF, sample_name, xm, ym2):
    path = os.getcwd() #current directory
    z = z[SF]
    zn = np.copy(z)
    
    xp = x*1000
    yp = y*1000

    con = np.linspace(0.1, 1, 9)
   
    cmap, vmin, vmax = FORCinel_colormap(zn) #runs FORCinel colormap
    
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
    plt.savefig(os.path.join(path+ os.sep,'processed_data','{}.pdf'.format(sample_name,SF)), bbox_inches='tight')
    plt.close()

    #zip up file 
    
    shutil.make_archive('processed_data', 'zip', 'processed_data')
    shutil.rmtree('processed_data') 

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
        X = find_fwhm(X, SFlist[i], name)
        i+=1 #0,1,2,3    
    return (X)
    
#FWHM function
def half_max_test(fwHu_c, fwRho_c, ym):
    arr_L = np.where(fwRho_c == ym)[0]
    L = arr_L[0]
    half_ym = ym/2. #half max
    b = L+1

    while (b < len(fwRho_c)):

        if(fwRho_c[b] < half_ym):
            
            break
        b = b + 1
    
    top = fwRho_c[b-1] - fwRho_c[b]
    bot = fwHu_c[b-1] - fwHu_c[b]
   
    mo_test = top/bot
    r0 = fwHu_c[b] + ((half_ym - fwRho_c[b])/mo_test)
   
    u = L-1

    while (u > 0): 
       
        if (fwRho_c[u] < half_ym):
            
            break
        u = u - 1
    

    m1 = (fwRho_c[u] - fwRho_c[u+1])/(fwHu_c[u] - fwHu_c[u+1])

    r1 = fwHu_c[u+1] + ((half_ym - fwRho_c[u+1])/m1)
  
    fwhm = r1 - r0
   
    return fwhm, r0, r1


def find_fwhm(X, SF, sample_name): 
    fwhmlist = X['fwhmlist']
    Rho = X['rho'] 

    Hu = X['Hu']

    indices = np.unravel_index(np.nanargmax(Rho[SF]),Rho[SF].shape)

    fwHu = []
    fwRho = []
    for i in range(len(Rho[SF])):
        fwHu.append(Hu[i][indices[1]]) 
        fwRho.append(Rho[SF][i][indices[1]])
        i+=1

    fwHu = np.array(fwHu)
    fwRho = np.array(fwRho)
    fwHu = fwHu[~np.isnan(fwHu)]
    fwRho = fwRho[~np.isnan(fwRho)] 
    r0 = 1
    r1 = -1

    
    loc_o = np.argmin(abs(fwHu))
    fwHu_f = fwHu[:loc_o] 
    fwRho_f = fwRho[:loc_o]

    loc_m = np.argmin(abs(fwRho_f))
    fwHu_c = fwHu[loc_m:(loc_o +(loc_o - loc_m))]
    fwRho_c = fwRho[loc_m:(loc_o +(loc_o - loc_m))]

    plt.plot(fwHu_c, fwRho_c)
    plt.show
    m_rho_a = np.sort(fwRho_c)
    i = 1
    while ((r0 >0) or (r1 < 0)): #opposte to FWHM crossing 0 
        ym = m_rho_a[-i]

        # find the two crossing points
        try:
  
            fwhm, r0, r1 = half_max_test(fwHu_c, fwRho_c, ym)

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
    plt.xlabel('$\mathrm{h_{s}}$ (mT)')
    plt.ylabel('FORC weighting')
    plt.legend()
    plt.title('Plot of the cross sections of the FWHM at each smoothing factor')

    plt.show
    X['fwhmlist'] = fwhmlist
    return(X)
    

def plot_fwhm_1(X):
    SFlist = X['SF_list']
    fwhmlist = X['fwhmlist']
    st_line_SFlist = []
    polyfwhm = []
    polySF = []

    maxSF1 = X['maxSF1']
    for i in range(maxSF1+1):
        st_line_SFlist.append(i)
        i +=1

    st_line_SFlist= np.array(st_line_SFlist)
    SFlist = np.array(SFlist)

    for i in range(len(fwhmlist)):
       if (fwhmlist[i] != 'Nan') and (fwhmlist[i] != 'NaN'): #add in fwhmlist[i] != nan
            polyfwhm.append(float(fwhmlist[i]))
            polySF.append(float(SFlist[i]))

    plt.scatter(polySF, polyfwhm)

    b, m = polyfit(polySF, polyfwhm, 1)
    X['b'] = b
    plt.title('Smoothing factor (SF) vs FWHM using SF 2-5')
    plt.xlabel('SF')
    plt.ylabel('FWHM')
    plt.plot(st_line_SFlist, b + m * st_line_SFlist, '-')

    plt.show 

    Hu = X['Hu']

    i=0

    X['fwhmlist'] = fwhmlist

    return(X)
    

def plot_fwhm(X):
    SFlist = X['SF_list']
    fwhmlist = X['fwhmlist']
    st_line_SFlist = []
    polyfwhm = []
    polySF = []

    maxSF1 = X['maxSF1']
    for i in range(maxSF1+1):
        st_line_SFlist.append(i)
        i +=1

    st_line_SFlist= np.array(st_line_SFlist)

    SFlist = np.array(SFlist)


    for i in range(len(SFlist)):
        if (fwhmlist[i] != 'che') and (fwhmlist[i] != 'Nan'): #remove value 

            polyfwhm.append(float(fwhmlist[i]))
            polySF.append(float(SFlist[i]))

    plt.scatter(polySF, polyfwhm)

    b, m = polyfit(polySF, polyfwhm, 1)
    X['b'] = b
    plt.xlabel('SF')
    plt.ylabel('FWHM')
    plt.title('SF versus FWHM plot with the accepted SFs')
    plt.plot(st_line_SFlist, b + m * st_line_SFlist, '-')

    plt.show 

    Hu = X['Hu']

    i=0
    for i in range(len(SFlist)):

        if (fwhmlist[i] == 'Nan') or (fwhmlist[i] == 'che'):

            fwhmlist[i] = float(m*SFlist[i] + b)


    X['fwhmlist'] = fwhmlist

    return(X)
    
    
def check_fwhm(X):
    SFlist = X['SF_list']
    answer = None
    answer2 = None
    fwhmlist = X['fwhmlist']
    maxSF1 = X['maxSF1']
    while answer not in ("yes", "no"):
        answer = input("Are any of the FWHM unreliable? Enter yes or no: ")
        if (answer == "yes"):
            sf_int = 0
            while (sf_int == 0):
                try:
                    sf_pick = (input("Which SF is unrealiable and needs to be removed?:" ))
                    sf_pick = int(sf_pick) #ask for interger and check it is an interger, if not ask again
                    sf_int = 1

                    if (sf_pick >= 2) and (sf_pick <= maxSF1):
                        sf_int = 1

                    else:
                        sf_int = 0
                        print('Not an interger between 2 and 5. Please input an interger between 2 and 5.')

                except ValueError:
                    print('Not an interger. Please input an interger between 2 and 5.')


            while answer2 not in ("yes", "no"):
                answer2 = input("Are any other FWHM unreliable? Enter yes or no: ")
        
                if (answer2 == "yes"):
                    sf_int2 = 0
                    while (sf_int2 == 0):
                        try:
                            sf_pick2 = (input("Which other SF is unrealiable and needs to be removed?:" ))
                            sf_pick2 = int(sf_pick2) #ask for interger and check it is an interger, if not ask again
                            sf_int2 = 1
 
                            if (sf_pick2 >= 2) and (sf_pick2 <= maxSF1):
                                sf_int2 = 1

                            else:
                                sf_int2 = 0
                                print('Not an interger between 2 and 5. Please input an interger between 2 and 5.')

                        except ValueError:
                            print('Not an interger. Please input an interger between 2 and 5.')
                        


                elif (answer2 == "no"):
                    print(answer2)

                
                elif (isinstance(answer2, str)):
                    print("Please enter yes or no.")
                    
                
            

            fwhmlist[sf_pick-2] = 'che' 
                        
              
            if (answer2 == "yes"):

                fwhmlist[sf_pick2-2] = 'che' 
            X['fwhmlist'] = fwhmlist

            X = plot_fwhm(X) 
            

        elif answer == "no":
        
            X = plot_fwhm(X) 
           
        elif (isinstance(answer, str)):
            print("Please enter yes or no.")
   
    fwhmlist = np.array(fwhmlist)
    X['fwhmlist'] = fwhmlist

    
    return(X)
        

def divide_mu0(X):
    mu0 = mu0=4*pi*1e-7
    X['Hc_mu'] = X['Hc']/mu0

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
    Hu_f = X['Hu_mu'].flatten() 
    Hc_f = X['Hc_mu'].flatten()
    Rho_f = X['rho'][SF].flatten() 
    
    #remove nan
    Hu_f = Hu_f[~np.isnan(Hu_f)]
    Hc_f = Hc_f[~np.isnan(Hc_f)]
    Rho_f = Rho_f[~np.isnan(Rho_f)]
    
    step_xi = np.nanmax(X['Hc_mu'])/181.
    step_yi = (np.nanmax(X['Hu_mu']) - np.nanmin(X['Hu_mu']))/146.

    # target grid to interpolate to
    xi = np.arange(0,np.nanmax(X['Hc_mu']),step_xi) 
    yi = np.arange(np.nanmin(X['Hu_mu']),np.nanmax(X['Hu_mu']),step_yi) 
    xi1,yi1 = np.meshgrid(xi,yi) 



    zi = griddata((Hc_f,Hu_f),Rho_f,(xi1,yi1),method='cubic') 

    X['xi1'] = xi1
    X['yi1'] = yi1
    X['zi_{}'.format(SF)] = zi
    return (X)
    
def inter_rho(xi_s_f, yi_s_f, zi_s_f, hys, i): # when call use xi_s etc, i is no. hysteron to test
    xi1_row = xi_s_f[0,:] 

    up_hc = xi1_row[xi1_row > hys[i,0]].min()   
    lo_hc = xi1_row[xi1_row < hys[i,0]].max() 

    up_hc_idx = list(xi1_row).index(up_hc) 
    lo_hc_idx = list(xi1_row).index(lo_hc)
    
    yi1_col = yi_s_f[:,0] 

    up_hi = yi1_col[yi1_col > hys[i,1]].min()   
    lo_hi = yi1_col[yi1_col < hys[i,1]].max() 

    up_hi_idx = list(yi1_col).index(up_hi) 
    lo_hi_idx = list(yi1_col).index(lo_hi)

    x_arr = np.array([xi_s_f[lo_hi_idx,lo_hc_idx], xi_s_f[up_hi_idx, lo_hc_idx], xi_s_f[up_hi_idx, up_hc_idx], xi_s_f[lo_hi_idx, up_hc_idx]])
    y_arr = np.array([yi_s_f[lo_hi_idx,lo_hc_idx], yi_s_f[up_hi_idx, lo_hc_idx], yi_s_f[up_hi_idx, up_hc_idx], yi_s_f[lo_hi_idx, up_hc_idx]])
    z_arr = np.array([zi_s_f[lo_hi_idx,lo_hc_idx], zi_s_f[up_hi_idx, lo_hc_idx], zi_s_f[up_hi_idx, up_hc_idx], zi_s_f[lo_hi_idx, up_hc_idx]])

   
    xarr_sum = np.sum(x_arr)
    xarr_has_nan = np.isnan(xarr_sum)
    yarr_sum = np.sum(y_arr)
    yarr_has_nan = np.isnan(yarr_sum)
    zarr_sum = np.sum(z_arr)
    zarr_has_nan = np.isnan(zarr_sum)    

    
    if (xarr_has_nan != True) and (yarr_has_nan != True) and (zarr_has_nan != True):
        f = interp2d(x_arr, y_arr, z_arr, kind='linear')
        hys[i,3] = f(hys[i,0], hys[i,1]) 
    else:
        hys[i,3] = -0.001
    


    return hys
    
    
def sym_FORC(X, SF):
    xi1 = X['xi1']
    yi1 = X['yi1']
    zi = X['zi_{}'.format(SF)] 
    yi_axis = np.copy(yi1)

    yi_axis = abs(yi_axis) - 0

    
    indices = np.unravel_index(np.nanargmin(yi_axis),yi_axis.shape)
   

    xi_s = np.copy(xi1)
    yi_s = np.copy(yi1)
    zi_s = np.copy(zi)
  
    x=1
    j=0

    while x < (len(xi1) - indices[0]): 

        j=0
        for j in range(len(xi_s[0])):


            find_mean = np.array([zi_s[indices[0]+x][j],zi_s[indices[0]-x][j]])
          
            find_mean = find_mean[~np.isnan(find_mean)]
            if (len(find_mean) > 0):
                zi_s[indices[0]+x][j] = np.mean(find_mean)
             
                zi_s[indices[0]-x][j] = zi_s[indices[0]+x][j]


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

    phi = acos(2*angle - 1)
    if(phi > (pi/2.)): 
        phi = pi-phi
    angle2 = random.random()
    phistatic = acos(2*angle2 - 1)
    if(phistatic > (pi/2.)):
        phistatic = pi-phistatic
    
    angle3 = random.random()
    thetastatic = 2*pi*angle3
    return phi, phistatic, thetastatic
    
def calc_hk_arrays(hys, num, V): 
    tm = V['tm']

    hc = np.copy(hys[:,0]) 
    tempt = 300.

    hf = ((hc/(sqrt(2)))**(0.54))*(10**(-0.52)) 
    phi = np.copy(hys[:,5])
    
    phitemp = (((np.sin(phi))**(2./3.))+((np.cos(phi))**(2./3.)))**(-3./2.) 

    gphitemp = (0.86 + (1.14*phitemp)) 
  
    hatmp = hc/(sqrt(2)) 

    ht = hf*(log(tm/tau)) 
    
    hktmp = hatmp +ht + (2*hatmp*ht+ht**2)**0.5 
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
                hstep = hstep*factor 

    hkphi = hktmpstore 
    hk = hkphi[:int(num)]/phitemp[:int(num)]

    hys[:int(num),9] = hk 

    return(hys)
    
    
def pop_hys(num_hys, X, V, SF): 
    #populate hysterons
    
    corr = X['sf_list_correct'][SF - 2]
    hys = np.zeros((num_hys,11)) 
    
    
    num_pop = num_hys/2
    
    xi_s_cut = X['xis']
    yi_s_cut = X['yis']
    zi_s_cut = X['zis_{}'.format(SF)] 
    maxHc = np.nanmax(xi_s_cut) 
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
        
        
        hys = inter_rho(xi_s_cut, yi_s_cut, zi_s_cut, hys, i) 
        hys[i,1] = hys[i,1]*corr
        hys[i,5], hys[i,6], hys[i,7] = hys_angles() #calc for half hys
        
        if ((hys[i,1]) <= (hys[i,0])) and (hys[i,3] >= 0) and (hys[i,3] <= 1): 
            i +=1 

    hys = calc_hk_arrays(hys, int(num_pop), V) # calc hk using arrrys 

    hys[:,4] = 1
 
    hys[:,8] = hys[:,5]*hys[:,4] 
    num_pop = int(num_pop)
    j=0
    for j in range(num_pop):
        hys[(j+num_pop),:] = hys[j,:]
        hys[j+num_pop,1] = -hys[j,1]
        j+=1


    return hys, num_pop

     
    
@jit(nopython = True)
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
 
    for i in range(num_hyss): 

        hc=(sqrt(2))*(hys[i,9])*beta 
       
        hcstore[i] = hc/(sqrt(2)) 

        hi = hys[i,1]*beta*blockper 

        histore[i] = hi/(sqrt(2)) 
 
        phitemp=((sin(hys[i,5])**(2./3.))+(cos(hys[i,5])**(2./3.)))**(-3./2.) 

        g=0.86+1.14*phitemp

        hf=((hys[i,9]**0.54))*(10**(-0.52))*temp/(300*beta) 
   
        hfstore[i] = hf 

        if (rate == 1): 

            r = (1./aconst)*(temp-tempmin)
  
            tm = (temp/r)*(1 - (temp/(curie_t+273)))/log((2*temp)/(tau*r)*((1. - (temp/(curie_t+273)))))      

            
            if (tm == 0.0): 
                tm = 60.0

        ht = (roottwohffield)*hf*(log(tm/tau))  

        bracket = 1-(2*ht*phitemp/hc)**(1/g)
        
        hiflipplus = hc*bracket+field*(roottwohffield) 

        hiflipminus=-hc*bracket+field*(roottwohffield) 
   
        if (hc >= (2*ht*phitemp)): 

            if ((hi > hiflipminus) and (hi < hiflipplus)):
              

                if ((blockg[i] == 0) or (blockg[i] == 2) or (blockg[i] == -2)): 
                    
                    if (hi >= (field*roottwohffield)): 
                        
                        blocktemp[i] = -1
                       
                    else:

                        blocktemp[i] = 1 
                      
                elif (blockg[i] == -1): 
                 
                    blocktemp[i] = -1
            

                elif (blockg[i] == 1):
  
                    blocktemp[i] = 1
                   
                else:
                    #write to screen, blockg[i]
                    print(blockg[i], blocktemp[i]) 
                    print('----', i)
                
            elif (hi >= hiflipplus):
               
                blocktemp[i] = -2
              
            else:
               
                blocktemp[i] = 2
        else: 

            if ((hi < hiflipminus) and (hi > hiflipplus)): 
               
                blocktemp[i] = 0
                if (heating == 1):
                
                    boltz[i] = 0.0

            else: 
                if (hi >= hiflipminus):

                    blocktemp[i] = -2
                else:

                    blocktemp[i] = 2

    return hfstore, histore, boltz, blocktemp  


@jit(nopython = True)
def block_val(hys, histore, hfstore, blocktemp, beta, num_hyss, boltz, blockg, field):
   
    i=0
    totalm = 0.0
    blockper = 0   
    for i in range(num_hyss):
        x = blocktemp[i]
        blockg[i] = x
        absblockg = abs(blockg[i])
        
        if (absblockg == 1): 
            if (boltz[i] < 0.00000001) and (boltz[i] > -0.000000001): #only zero

                boltz[i] = tanh((field - histore[i])/hfstore[i])
  
        if (blockg[i] == -2):
         
            moment = -0 
        elif (blockg[i] == 2):

            moment = 0
        else:

            moment = blockg[i] 

       

        totalm = totalm + abs(moment)*abs(cos(hys[i,5]))*hys[i,3]*beta*(boltz[i]) 


        blockper=blockper+abs(moment)*hys[i,3]*1.0 
        i+=1

        
    blockper=blockper/(1.0*np.sum(hys[:,3]))

    return blockper, totalm, boltz, blockg
    

def blockfind(temp, field, afswitch, V, X): 

    hys = V['hys']
    num_hyss = V['num_hyss']
    hcstore = V['hcstore']
    histore = V['histore']
    beta = V['beta']
    rate = V['rate']
    aconst = V['aconst']
    tempmin = V['tempmin']
    
    heating = V['heating']
    
    tm = V['tm'] 

    hfstore = np.zeros(num_hyss)

    blockper = V['blockper'] 

    blocktemp = V['blocktemp'] 
    boltz = V['boltz']
    blockg = V['blockg']

    curie_t = V['curie_t']

    totalm = V['totalm']


    var_1 = np.array((num_hyss, beta, blockper, temp, aconst, curie_t, rate, tempmin, field, tm, heating))

    hfstore, histore, boltz, blocktemp = block_loc(var_1, hys, blockg, boltz)  
    
    blockper, totalm, boltz, blockg = block_val(hys, histore, hfstore, blocktemp, beta, num_hyss, boltz, blockg, field)
    
    V['blockper'] = blockper
    V['blocktemp'] = blocktemp
    V['boltz'] = boltz
    V['blockg'] = blockg
    V['totalm'] = totalm
    V['tm'] = tm

    return(V) 
      
  
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
        

        i=0
        for i in range(len(list_ms)):
            if (list_ms[i] == 0):
                curie_idx = i-1
                break
            i+=1
        curie_t = temp_ms[curie_idx] 
        
        V['curie_t'] = curie_t

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

   
    m_ratio = np.zeros_like(rev_mag_lab)
    rev = 0
    for rev in range(len(rev_mag_lab)):
        if (rev_mag_lab[rev] > 1E-10):
            m_ratio[rev] = rev_mag_nat[rev]/rev_mag_lab[rev]    
        else:
            m_ratio[rev] = 1.
            


    for i in range(len(m_ratio)):
        if (rev_mag_nat[i] < (0.002*(np.nanmax(rev_mag_nat)))) and  (rev_mag_lab[i] < (0.002*np.nanmax(rev_mag_lab))): 
            m_ratio[i] = 1
    

  
    m_ratio_av = np.zeros_like(m_ratio) 

    for i in range(len(m_ratio)):

        if (i < 3):
            lb = 0
        else:
            lb = i - 3
        if (i> len(m_ratio) -3): 
            ub = len(m_ratio)
        else:
            ub = i+3

        m_ratio_av[i] = np.mean(m_ratio[lb:ub+1])


    V['m_ratio_av'] = m_ratio_av
    return(V)
    

def fix_files(V):
    track2 = V['track2']
    temp_c = V['temp_h'][0][:track2] - 273
    m_ratio_av = V['m_ratio_av']
    files_in = []
    file_name_only = []
    path = os.getcwd() #current directory 

    try:
        os.mkdir('processed_data')
    except:
        print('This step has already been carried out and a processed_data folder exists. To add more files, restart the kernel and re-run the previous cells too.')
    print('Corrected Thellier files:')
    for root, dir, file in os.walk(path):
        for f in file:

            if re.match('.*.tdt', f):
                files_in.append(os.path.join(root + os.sep,f))
                file_name_only.append(f)
    for i in range(len(files_in)):
        name_in = []
        intensity_in = []
        temp_in = []
        dec_in = []
        inc_in = []
        with open(files_in[i]) as my_file:
            count = 0
            for line in my_file:

                if (count == 0):
                    header = line.split()
                    top_line = header

                elif (count == 1):
                    header2 = line.split()

                    col1 = header2[0]
                    col2 = header2[1]
                    col3 = header2[2]
                    col4 = header2[3] 
                    col5 = header2[4] 

                else:

                    line_v = line.split()

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



            j=0
            for j in range(len(temp_in_a)):
                if (str(temp_in_a[j]).endswith('.1')) or (str(temp_in_a[j]).endswith('.2')):


                    temp_in_field = temp_in_a[j]

                    x = j
                    while (str(temp_in_a[x]).endswith('.0') == False):
                    
                        loc_out = x
                        x-= 1
                    loc_out = x
                 
                    temp_out_field = float(temp_in_a[loc_out])

                    cf_loc1 = (np.abs((temp_c - temp_in_field))).argmin() 
                    if ((temp_c[cf_loc1]) > temp_in_field):
                        cf_loc2 = cf_loc1-1
                    else:
                        cf_loc2 = cf_loc1+1
                    if (cf_loc1 < len(m_ratio_av)) and (cf_loc2 < len(m_ratio_av)):
                        cf = np.interp(temp_in_field, [temp_c[cf_loc1], temp_c[cf_loc2]], [m_ratio_av[cf_loc1], m_ratio_av[cf_loc2]])
                    else:
                        cf = m_ratio_av[cf_loc1]

                    in_field = intensity_in_a[j]

                    
                    out_field = intensity_out[loc_out]
                    intensity_out[j] = sqrt(((cf*xt[j]*in_field + xt[loc_out]*out_field*(1-cf))**2) + ((cf*yt[j]*in_field+yt[loc_out]*out_field*(1-cf))**2) + ((cf*zt[j]*in_field+zt[loc_out]*out_field*(1-cf))**2))
                j+=1


            file_out_name  = os.path.join(path+ os.sep,'processed_data',file_name_only[i][:(len(file_name_only[i])-4)])
            print(file_name_only[i][:(len(file_name_only[i])-4)]+'_U.tdt')
            f = open('{}_U.tdt'.format(file_out_name), 'w')
            f.write('Thellier-tdt' + "\n")
            f.write(str(col1) + "\t" + str(col2) + "\t" + str(col3) + "\t" + str(col4) + "\t" + str(col5) + '\n') 
            k=0
            for k in range(len(inc_in)):
                f.write(name_in[k] + "\t" + str("{:.1f}".format(temp_in[k])) +  "\t" + str("{:.12f}".format(intensity_out[k])) + "\t" + str(dec_in[k]) + "\t" + str(inc_in[k]) + "\n")
            f.close()
            
    return
    
def TRM_acq(X, V, curve):

    SF = X['SF']
    
    if (curve == True):
        temp_ms = V['temp_ms']
        list_ms = V['list_ms']
        curie_t = V['curie_t']
    else:
        curie_t = V['curie_t']
          
    V['heating'] = 0.
    mu0 = 4*pi*1e-7
    kb = 1.3806503e-23
    tau = 10e-9
    roottwohffield = 2**(0.5)  

    afone = 1
    afzero = 0

    blockper = 0.0
    V['blockper'] = blockper

    num_hyss = 300000
    V['num_hyss'] = num_hyss

    tempmax = float(V['curie_t'] + 273)
    tempmin=300 
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

        V['msR'] = ms0 
        beta_a = list_ms/(ms0) 
        V['beta_a'] = beta_a    
    

    blockg = np.zeros(num_hyss) 
    boltz = np.zeros(num_hyss)
    blocktemp = np.zeros(num_hyss) 

    V['blocktemp'] = blocktemp
    V['boltz'] = boltz
    V['blockg'] = blockg
    V['totalm'] = 0


    i=0 
    temp=300
    tempt=300

    tm = 0.2 
    V['tm'] = tm


    hys1, num_pop = pop_hys(num_hyss, X, V, SF)   

    ct = 0
    for ct in range(len(V['variable_change'])):
        field = (50E-6)/mu0
        ifield = ct
        ac = V['variable_change'][ct]

        blockg = np.zeros(num_hyss) 
        boltz = np.zeros(num_hyss)
        blocktemp = np.zeros(num_hyss) 

        V['blocktemp'] = blocktemp
        V['boltz'] = boltz
        V['blockg'] = blockg
        V['totalm'] = 0


        i=0 
        temp=300
        tempt=300

        tm = 0.2 
        V['tm'] = tm

        temp = tempmax
        rate = 1 
        V['rate'] = rate


        afswitch = afzero

        V['hys'] = copy.deepcopy(hys1)
       
        tm = 0

        V['tm'] = tm
        track = 0
        temp = curie_t + 273 - 1

        while (temp > tempmin):
            aconst=(-ac*60.0*60.0)/(log(0.01*(tempmin)/(tempmax-tempmin))) #line 708 
            V['aconst'] = aconst

            if (curve == True):
                beta_loc1 = (np.abs((temp_ms - temp))).argmin()
            
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

            V = blockfind(temp, field, afzero, V, X) 

            V['trm_a'][ifield][track] = V['totalm']
            V['temp_a'][ifield][track] = temp

            track +=1
           

            tempstep1 = tempstep*2
            temp = temp - tempstep1
 
        rate = 0
        V['rate'] = rate
        tm = 60
        V['tm'] = tm
        temp = tempmin + 0.1 
   
        if (curve == True):
            beta_loc1 = (np.abs((temp_ms - temp))).argmin()
            if (beta_loc1 != 0):
                if ((temp_ms[beta_loc1]) > temp) and (temp_ms[beta_loc1-1] < temp): 
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

        fieldzero = 0.0
        V = blockfind(temp, fieldzero, afzero, V, X) 
        
        V['trm_a'][ifield][track] = V['totalm'] 
        V['temp_a'][ifield][track] = temp
        track+=1


        track2 = 0
        temp = 300.1
        
        rate = 1.
        V['rate'] = rate 
        
        ac = float(V['variable_change'][0]) 
        V['heating'] = 1.

        while (temp < (curie_t + 273)):
            aconst=(-ac*60.0*60.0)/(log(0.01*(tempmin)/(tempmax-tempmin))) 
            V['aconst'] = aconst
      
            if (curve == True):
                msR = V['ms_R']
                beta_loc1 = (np.abs((temp_ms - temp))).argmin()
        
                if (beta_loc1 != 0):
                    if ((temp_ms[beta_loc1]) > temp) and (temp_ms[beta_loc1-1] < temp): 
                        beta_loc2 = beta_loc1-1
                        beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])  
                    elif (beta_loc1 != (len(temp_ms) -1)):
                        beta_loc2 = beta_loc1+1
                        beta_u = np.interp(temp, [temp_ms[beta_loc1], temp_ms[beta_loc2]], [beta_a[beta_loc1], beta_a[beta_loc2]])   
                    elif (beta_loc1 == (len(temp_ms)-1)):
                        beta_u = beta_a[beta_loc1] 
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
         
            V['heating'] = 1.
            V = blockfind(temp, field, afzero, V, X) 
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
            V = blockfind(300.1, field, afzero, V, X)     
         
            V['heat'][ifield][track2] = V['totalm']
            V['temp_h'][ifield][track2] = temp

            track2 +=1


          
            tempstep1 = tempstep*5
            temp = temp + tempstep1


        
        temp = tempmin
        

        j = ifield 
    

        V['heating'] = 0.
        ifield = ifield +1 
        V['aconst'] = aconst

        V['ifield'] = ifield
        V['fields'] = fields
        V['track'] = track
        V['track2'] = track2        
       
        ct +=1
        
        
        
        
    return    
    
    
    
    