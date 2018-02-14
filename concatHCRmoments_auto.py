# **** concatHCRmoments ****
#
# This script combines multiple CF/Radial output files from the HIAPER Cloud Radar (HCR) into a single output netCDF file. 
#    The most comonly required variables for analysis are concatenated into single arrays, and the inclusion of additional variables 
#    is easy enough to do if desired.
# 
# Currently this has only been tested with 10hz moments files, though combining the 100hz moments files is not entirely out of the question.
# 
# Written by Dan Stechman
# University of Illinois at Urbana-Champaign


from netCDF4 import Dataset
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import xarray as xr
from glob import glob
import time
from datetime import datetime as dt
import argparse

warnings.filterwarnings("ignore",category=RuntimeWarning)

scriptStrtT = dt.now()

parser = argparse.ArgumentParser()
parser.add_argument("flight", help="research flight number and date in the form: RF##_YYYYMMDD")
parser.add_argument("-s", "--startT", help="string indicating start time of data to be concatenated (in YYYYMMDD_HHMMSS)", default="beg")
parser.add_argument("-e", "--endT", help="string indicating end time of data to be concatenated (in YYYYMMDD_HHMMSS)", default="end")
parser.add_argument("-d", "--dataPath", help="base path of data location. This is where input and output data are stored", default="/Volumes/SOCRATES_1/")

args = parser.parse_args()


# This variable should match the name of the parent directory containing
#    the flight data for the given mission. Used in defining file input 
#    and output names.
flight = args.flight


# Specify start and end times to include in concatenated dataset
#    startT and endT should be strings of the format 'YYYYmmdd_HHMMSS'
#    Otherwise, set startT to 'beg' to use the first file in the whole flight directory
#    and/or set endT to 'end' to use the last file in the whole flight directory
startT = args.startT
endT = args.endT

# Specify the parent path of the data directory where input/output are stored
dataPath = args.dataPath


# Specify the location/name(s) of the input files and the output file
files = sorted(glob(dataPath + flight + '/HCR/cfradial/10hz_moments/cfrad.*HCR*.nc'))
outFileName = dataPath + flight + '/HCR/' + flight + '_concat-HCR-moments.nc'



# These are the number of gates nearest the aircraft with negative gate range values (we throw these out)
#    **I've seen this number vary between projects, but seems consistent so far for SOCRATES at least
rGateStrt = 12


# **** Data File ID ****
# Determine which data files will provide data between the user-defined start and end times
if startT == 'beg':
    strtFIx = 0
else:
    # Search string designed to grab cfradial files with the desired start times
    strtStr = '.' + startT
    
    # Get the index of the first occurence of cfradial file containing the 
    #     start string in its file name  
    strtFIx = ([ix for ix, f in enumerate(files) if strtStr in f])[0]
    
        
if endT == 'end':
    endFIx = len(files)-1
else:
    # Search string designed to grab cfradial files with the desired end times
    endStr = 'to_' + endT
    
    # Get the index of the first occurence of cfradial file containing the 
    #     end string in its file name
    endFIx = ([ix for ix, f in enumerate(files) if endStr in f])[0]
    


# Create an array of file indices (from our file listing) to loop over
# rFileIx = np.arange(210,220)
rFileIx = np.arange(strtFIx,endFIx)

print('{}    Starting HCR cfradial concatenation for {}...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S'),flight))

# **** Define output variables ****
# Loop through all of the input files and add up the total number of points in the time dimension and determine the number 
#    of gates (vertical dimension) after removing gates with negative ranges.
# Then, create empty (zero-filled) arrays of the appropriate sizes for out output.
print('{}    Determining output dimensions...'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))
totalTime = 0;
for rFile in rFileIx:
    radData = xr.open_dataset(files[rFile])
    totalTime += radData.dims['time']
totalRng = radData.dims['range'] - rGateStrt

print('\tTotal number of points in time dimension: {}'.format(totalTime))
print('\tTotal number of gates (in vertical dimension): {}\n'.format(totalRng))

time1d_all = np.zeros((totalTime,),dtype='datetime64[ns]')
radElev_all = np.zeros((totalTime,))
alt1d_all = np.zeros((totalTime,))
alt_all = np.zeros((totalTime,totalRng))
time_all = np.zeros((totalTime,totalRng),dtype='datetime64[ns]')
dbz_all = np.zeros((totalTime,totalRng))
vel_all = np.zeros((totalTime,totalRng))
width_all = np.zeros((totalTime,totalRng))
snrhx_all = np.zeros((totalTime,totalRng))
snrvc_all = np.zeros((totalTime,totalRng))
dbmhx_all = np.zeros((totalTime,totalRng))
dbmvc_all = np.zeros((totalTime,totalRng))
ncp_all = np.zeros((totalTime,totalRng))
ldr_all = np.zeros((totalTime,totalRng))


# **** Concatenate Variables Across All Files ****
# Loop through the files covering the requested time period and fill out output arrays.
#     Also, create modified time and altitude arrays which need to be 2-dimensional for plotting.
#     The altitude array is modified using the gate range from the aircraft to yield ground-relative altitudes for each gate.
print('{}    Beginning file concatenation for {}...'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S'),flight))
numFiles = len(rFileIx)
fileCount = 1
strtIx = 0
for rFile in rFileIx:
    print('\tAppending file {} of {}...'.format(fileCount,numFiles))
    
    # Open the current radar data file
    radData = xr.open_dataset(files[rFile])
    
    # Get length of time dimension of current file and define the end index
    #    of the array slice to place data within the output arrays
    endIx = radData.dims['time'] + strtIx
        
    # Pull out the dimension variables
    gateRange_1d = radData['range'].data[rGateStrt:]
    alt_1d = radData['altitude'].data
    time_1d = radData['time'].data
    
    # Get the beam elevation angle variable (used to determine upward/downward pointing)
    radElev_1d = radData.elevation.data
    
    # HCR elevation angle is positive when pointing upward
    # Adjust the gateRange variable if this is the case to properly initialize
    #    the ground-relative altitude array
    if np.all(radElev_1d > 0):
        gateRange_1d *= -1
    
    # Create arrays of gateRange and gate altitude matching dimensions of 2-D variables
    gateRange_2d = np.transpose(np.tile(gateRange_1d,(len(time_1d),1)))
    time_2d = np.tile(time_1d,(len(gateRange_1d),1))
    alt_2d = np.tile(alt_1d,(len(gateRange_1d),1))

    # Add data from current file into slice of output arrays
    alt_all[strtIx:endIx,:] = np.transpose(alt_2d - gateRange_2d) # ground-relative altitude MSL of each gate
    time_all[strtIx:endIx,:] = np.transpose(time_2d)
    
    time1d_all[strtIx:endIx] = time_1d[:]
    radElev_all[strtIx:endIx] = radElev_1d[:]
    
    alt1d_all[strtIx:endIx] = alt_1d[:]

    dbz_all[strtIx:endIx,:] = radData['DBZ'].data[:,rGateStrt:]
    vel_all[strtIx:endIx,:] = radData['VEL'].data[:,rGateStrt:]
    width_all[strtIx:endIx,:] = radData['WIDTH'].data[:,rGateStrt:]
    snrhx_all[strtIx:endIx,:] = radData['SNRHX'].data[:,rGateStrt:]
    snrvc_all[strtIx:endIx,:] = radData['SNRVC'].data[:,rGateStrt:]
    dbmhx_all[strtIx:endIx,:] = radData['DBMHX'].data[:,rGateStrt:]
    dbmvc_all[strtIx:endIx,:] = radData['DBMVC'].data[:,rGateStrt:]
    ncp_all[strtIx:endIx,:] = radData['NCP'].data[:,rGateStrt:]
    ldr_all[strtIx:endIx,:] = radData['LDR'].data[:,rGateStrt:]
        
    # Move our starting index for the output array to immediately
    #    after the end of the current data slice
    strtIx = endIx
    
    fileCount += 1


# **** Write concatenated data out to NetCDF ****
print('\n{}    Concatenation complete. Beginning netCDF write...'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))
# Create a netCDF file to hold our output
rootGrp = Dataset(outFileName,'w',format='NETCDF4')
rootGrp.set_fill_on()

# Define the netCDF dimensions
gateRng = rootGrp.createDimension('gateRng',totalRng)
time1d = rootGrp.createDimension('time1d',totalTime)

# Create out output variable instances within the file and
#    define metadata for each as needed
TIME = rootGrp.createVariable('time1d','f8',('time1d',))
TIME.long_name = 'Timestamp'
TIME.units = 'UTC YYYYMMDDTHHMMSS.f'
TIME.coordinates = 'time1d'

GRNG = rootGrp.createVariable('gateRng','f4',('gateRng',),fill_value=np.nan)
GRNG.long_name = 'Distance from radar to center of each range gate'
GRNG.units = 'm'
GRNG.coordinates = 'gateRng'

TIME2D = rootGrp.createVariable('time2d','f8',('time1d','gateRng'))
TIME2D.long_name = 'Timestamp for every gate/time'
TIME2D.units = 'UTC YYYYMMDDTHHMMSS.f'
TIME2D.coordinates = 'time1d gateRng'

GALT2D = rootGrp.createVariable('gateAlt2d','f4',('time1d','gateRng'),fill_value=np.nan)
GALT2D.long_name = 'Altitude of each gate above MSL'
GALT2D.units = 'm'
GALT2D.coordinates = 'time1d gateRng'

ALT = rootGrp.createVariable('planeAlt','f4',('time1d',),fill_value=np.nan)
ALT.long_name = 'Aircraft altitude above MSL'
ALT.units = 'm'
ALT.coordinates = 'time1d'

DBZ = rootGrp.createVariable('DBZ','f4',('time1d','gateRng'),fill_value=np.nan)
DBZ.long_name = 'reflectivity'
DBZ.units = 'dBZ'
DBZ.sampling_ratio = 1.000000
DBZ.grid_mapping = 'grid_mapping'
DBZ.coordinates = 'time1d gateRng'

VEL = rootGrp.createVariable('VEL','f4',('time1d','gateRng'),fill_value=np.nan)
VEL.long_name = 'doppler_velocity'
VEL.units = 'm/s'
VEL.sampling_ratio = 1.000000
VEL.grid_mapping = 'grid_mapping'
VEL.coordinates = 'time1d gateRng'

WIDTH = rootGrp.createVariable('WIDTH','f4',('time1d','gateRng'),fill_value=np.nan)
WIDTH.long_name = 'spectrum_width'
WIDTH.units = 'm/s'
WIDTH.sampling_ratio = 1.000000
WIDTH.grid_mapping = 'grid_mapping'
WIDTH.coordinates = 'time1d gateRng'

LDR = rootGrp.createVariable('LDR','f4',('time1d','gateRng'),fill_value=np.nan)
LDR.long_name = 'linear_depolarization_ratio (VV to VH)'
LDR.units = 'dB'
LDR.sampling_ratio = 1.000000
LDR.grid_mapping = 'grid_mapping'
LDR.coordinates = 'time1d gateRng'

NCP = rootGrp.createVariable('NCP','f4',('time1d','gateRng'),fill_value=np.nan)
NCP.long_name = 'normalized_coherent_power'
NCP.units = ''
NCP.sampling_ratio = 1.000000
NCP.grid_mapping = 'grid_mapping'
NCP.coordinates = 'time1d gateRng'

SNRVC = rootGrp.createVariable('SNRVC','f4',('time1d','gateRng'),fill_value=np.nan)
SNRVC.long_name = 'signal_to_noise_ratio (VV)'
SNRVC.units = 'dB'
SNRVC.sampling_ratio = 1.000000
SNRVC.grid_mapping = 'grid_mapping'
SNRVC.coordinates = 'time1d gateRng'

SNRHX = rootGrp.createVariable('SNRHX','f4',('time1d','gateRng'),fill_value=np.nan)
SNRHX.long_name = 'signal_to_noise_ratio (VH)'
SNRHX.units = 'dB'
SNRHX.sampling_ratio = 1.000000
SNRHX.grid_mapping = 'grid_mapping'
SNRHX.coordinates = 'time1d gateRng'

DBMVC = rootGrp.createVariable('DBMVC','f4',('time1d','gateRng'),fill_value=np.nan)
DBMVC.long_name = 'log power (VV)'
DBMVC.units = 'dBm'
DBMVC.sampling_ratio = 1.000000
DBMVC.grid_mapping = 'grid_mapping'
DBMVC.coordinates = 'time1d gateRng'

DBMHX = rootGrp.createVariable('DBMHX','f4',('time1d','gateRng'),fill_value=np.nan)
DBMHX.long_name = 'log power (VH)'
DBMHX.units = 'dBm'
DBMHX.sampling_ratio = 1.000000
DBMHX.grid_mapping = 'grid_mapping'
DBMHX.coordinates = 'time1d gateRng'

RELV = rootGrp.createVariable('elevation','f4',('time1d',),fill_value=np.nan)
RELV.long_name = 'Radar beam elevation (positive is upwards [plane-relative])'
RELV.units = 'degrees'
RELV.coordinates = 'time1d'


# Define global attributes
rootGrp.description = 'Concatenated HIAPER Cloud Radar data'
rootGrp.flight = flight
rootGrp.history = 'Created ' + time.asctime(time.gmtime()) + ' UTC'
rootGrp.firstFile = files[strtFIx]
rootGrp.lastFile = files[endFIx]

## Write data into netCDF variables
TIME[:] = time1d_all.astype(np.float64)
GRNG[:] = gateRange_1d
TIME2D[:] = time_all.astype(np.float64)
GALT2D[:] = alt_all
ALT[:] = alt1d_all
DBZ[:] = dbz_all
VEL[:] = vel_all
WIDTH[:] = width_all
LDR[:] = ldr_all
NCP[:] = ncp_all
SNRVC[:] = snrvc_all
SNRHX[:] = snrhx_all
DBMVC[:] = dbmvc_all
DBMHX[:] = dbmhx_all
RELV[:] = radElev_all

# Close the output file
rootGrp.close()

print('\nTotal script run time: {}'.format(dt.now() - scriptStrtT))