# **** concatHSRLprelimData ****
# This script combines multiple "preliminary_data" output files from the High Spectral Resolution Lidar (HSRL) into 
#    a single output netCDF file. These input data files were created in the field by HSRL scientists and include 
#    various corrections. The most comonly required variables for analysis are concatenated into single arrays, and 
#    the inclusion of additional variables is easy enough to do if desired.
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

# if not args.runCIP and not args.runPIP:
#     sys.exit('You must specify at least one probe to process (--runCIP and/or --runPIP)')

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
files = sorted(glob(dataPath + flight + '/HSRL/preliminary_data/*GVHSRL*.nc'))
outFileName = dataPath + flight + '/HSRL/' + flight + '_concat-HSRL-prelimData.nc'



# **** Data File ID ****
# Determine which data files will provide data between the user-defined start and end times
# Create an array of file indices (from our file listing) to loop over
# lFileIx = np.arange(0,3)
lFileIx = np.arange(0,len(files))

print('{}    Starting HSRL preliminary_data concatenation for {}...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S'),flight))

# **** Define output variables ****
# Loop through all of the input files and add up the total number of points in the time dimension 
#    and determine the number of gates (vertical dimension) after removing gates with negative ranges.
# Then, create empty (zero-filled) arrays of the appropriate sizes for out output.
print('{}    Determining output dimensions...'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))
totalTime = 0;
for lFile in lFileIx:
    lidarData = xr.open_dataset(files[lFile],decode_times=False)
    totalTime += lidarData.dims['time']
totalAlt = lidarData.dims['range']

print('\tTotal number of points in time dimension: {}'.format(totalTime))
print('\tTotal number of gates (in vertical dimension): {}\n'.format(totalAlt))


# **** Initialize empty output arrays of appropriate size ****
time1d_all = np.zeros((totalTime,))
gateAlt1d_all = np.zeros((totalAlt,))
telescopeDir_all = np.zeros((totalTime,))
gateAlt2d_all = np.zeros((totalTime,totalAlt))
time2d_all = np.zeros((totalTime,totalAlt))

planeAlt_all = np.zeros((totalTime,))

AerosolBackscatterCoefficient_all = np.zeros((totalTime,totalAlt))
AerosolBackscatterCoefficientVariance_all = np.zeros((totalTime,totalAlt))
AerosolBackscatterCoefficientMask_all = np.zeros((totalTime,totalAlt))
ParticleDepolarization_all = np.zeros((totalTime,totalAlt))
ParticleDepolarizationVariance_all = np.zeros((totalTime,totalAlt))
ParticleDepolarizationMask_all = np.zeros((totalTime,totalAlt))
VolumeDepolarization_all = np.zeros((totalTime,totalAlt))
VolumeDepolarizationVariance_all = np.zeros((totalTime,totalAlt))
VolumeDepolarizationMask_all = np.zeros((totalTime,totalAlt))
BackscatterRatio_all = np.zeros((totalTime,totalAlt))
BackscatterRatioVariance_all = np.zeros((totalTime,totalAlt))
BackscatterRatioMask_all = np.zeros((totalTime,totalAlt))
MolecularBackscatterCoefficient_all = np.zeros((totalTime,totalAlt))
MolecularBackscatterCoefficientVariance_all = np.zeros((totalTime,totalAlt))
LowGainTotalBackscatterChannel_all = np.zeros((totalTime,totalAlt))
LowGainTotalBackscatterChannelVariance_all = np.zeros((totalTime,totalAlt))
HighGainTotalBackscatterChannel_all = np.zeros((totalTime,totalAlt))
HighGainTotalBackscatterChannelVariance_all = np.zeros((totalTime,totalAlt))
MolecularBackscatterChannel_all = np.zeros((totalTime,totalAlt))
MolecularBackscatterChannelVariance_all = np.zeros((totalTime,totalAlt))
CrossPolarizationChannel_all = np.zeros((totalTime,totalAlt))
CrossPolarizationChannelVariance_all = np.zeros((totalTime,totalAlt))
MergedCombinedChannel_all = np.zeros((totalTime,totalAlt))
MergedCombinedChannelVariance_all = np.zeros((totalTime,totalAlt))
MergedCombinedChannelMask_all = np.zeros((totalTime,totalAlt))
AerosolExtinctionCoefficient_all = np.zeros((totalTime,totalAlt))
AerosolExtinctionCoefficientVariance_all = np.zeros((totalTime,totalAlt))
AerosolExtinctionCoefficientMask_all = np.zeros((totalTime,totalAlt))


# **** Concatenate Variables Across All Files ****
# Loop through the files covering the requested time period and fill out output arrays.
# Also, create modified time and altitude arrays which need to be 2-dimensional for plotting.
#    The altitude array is modified using the gate range from the aircraft to yield 
#    ground-relative altitudes for each gate.
print('{}    Beginning file concatenation for {}...'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S'),flight))
numFiles = len(lFileIx)
fileCount = 1
strtIx = 0
for lFile in lFileIx:
    print('\tAppending file {} of {}...'.format(fileCount,numFiles))
    
    fDate = dt.date(dt.strptime(files[lFile][-30:-22],'%Y%m%d')) # Get the current date as a date object
    
    # Open the current lidar data file
    lidarData = xr.open_dataset(files[lFile],decode_times=False)
    
    # Get length of time dimension of current file and define the end index
    #    of the array slice to place data within the output arrays
    endIx = lidarData.dims['time'] + strtIx
        
    # Pull out the dimension variables
    gateAlt_1d = lidarData['range'].data
    time_1d_hhmmss = [dt.time(dt.utcfromtimestamp(t_sec)) for t_sec in lidarData['time'].data]
    
    # Convert time to an array of timestamps (use the dt.from_timestamp() function when pulling back out)
    time_1d = [dt.timestamp(dt.combine(fDate, t)) for t in time_1d_hhmmss]
    
    # Get the telescope pointing direction
    telescopeDir = lidarData['TelescopeDirection'].data # 1 = upward, 0 = downward

    
    # Create arrays of gate altitude and time matching dimensions of 2-D variables
    gateAlt_2d = np.tile(gateAlt_1d,(len(time_1d),1))
    time_2d = np.transpose(np.tile(time_1d,(len(gateAlt_1d),1)))
    

    # Add data from current file into slice of output arrays
    gateAlt2d_all[strtIx:endIx,:] = gateAlt_2d # altitude MSL of each gate
    time2d_all[strtIx:endIx,:] = time_2d
    
    time1d_all[strtIx:endIx] = time_1d[:]
    telescopeDir_all[strtIx:endIx] = telescopeDir[:]
    
    planeAlt_all[strtIx:endIx] = lidarData['GGALT'].data

    AerosolBackscatterCoefficient_all[strtIx:endIx,:] = lidarData['Aerosol_Backscatter_Coefficient'].data
    AerosolBackscatterCoefficientVariance_all[strtIx:endIx,:] = lidarData['Aerosol_Backscatter_Coefficient_variance'].data
    AerosolBackscatterCoefficientMask_all[strtIx:endIx,:] = lidarData['Aerosol_Backscatter_Coefficient_mask'].data
    ParticleDepolarization_all[strtIx:endIx,:] = lidarData['Particle_Depolarization'].data
    ParticleDepolarizationVariance_all[strtIx:endIx,:] = lidarData['Particle_Depolarization_variance'].data
    ParticleDepolarizationMask_all[strtIx:endIx,:] = lidarData['Particle_Depolarization_mask'].data
    VolumeDepolarization_all[strtIx:endIx,:] = lidarData['Volume_Depolarization'].data
    VolumeDepolarizationVariance_all[strtIx:endIx,:] = lidarData['Volume_Depolarization_variance'].data
    VolumeDepolarizationMask_all[strtIx:endIx,:] = lidarData['Volume_Depolarization_mask'].data
    BackscatterRatio_all[strtIx:endIx,:] = lidarData['Backscatter_Ratio'].data
    BackscatterRatioVariance_all[strtIx:endIx,:] = lidarData['Backscatter_Ratio_variance'].data
    BackscatterRatioMask_all[strtIx:endIx,:] = lidarData['Backscatter_Ratio_mask'].data
    MolecularBackscatterCoefficient_all[strtIx:endIx,:] = lidarData['Molecular_Backscatter_Coefficient'].data
    MolecularBackscatterCoefficientVariance_all[strtIx:endIx,:] = lidarData['Molecular_Backscatter_Coefficient_variance'].data
    LowGainTotalBackscatterChannel_all[strtIx:endIx,:] = lidarData['Low_Gain_Total_Backscatter_Channel'].data
    LowGainTotalBackscatterChannelVariance_all[strtIx:endIx,:] = lidarData['Low_Gain_Total_Backscatter_Channel_variance'].data
    HighGainTotalBackscatterChannel_all[strtIx:endIx,:] = lidarData['High_Gain_Total_Backscatter_Channel'].data
    HighGainTotalBackscatterChannelVariance_all[strtIx:endIx,:] = lidarData['High_Gain_Total_Backscatter_Channel_variance'].data
    MolecularBackscatterChannel_all[strtIx:endIx,:] = lidarData['Molecular_Backscatter_Channel'].data
    MolecularBackscatterChannelVariance_all[strtIx:endIx,:] = lidarData['Molecular_Backscatter_Channel_variance'].data
    CrossPolarizationChannel_all[strtIx:endIx,:] = lidarData['Cross_Polarization_Channel'].data
    CrossPolarizationChannelVariance_all[strtIx:endIx,:] = lidarData['Cross_Polarization_Channel_variance'].data
    MergedCombinedChannel_all[strtIx:endIx,:] = lidarData['Merged_Combined_Channel'].data
    MergedCombinedChannelVariance_all[strtIx:endIx,:] = lidarData['Merged_Combined_Channel_variance'].data
    MergedCombinedChannelMask_all[strtIx:endIx,:] = lidarData['Merged_Combined_Channel_mask'].data
    AerosolExtinctionCoefficient_all[strtIx:endIx,:] = lidarData['Aerosol_Extinction_Coefficient'].data
    AerosolExtinctionCoefficientVariance_all[strtIx:endIx,:] = lidarData['Aerosol_Extinction_Coefficient_variance'].data
    AerosolExtinctionCoefficientMask_all[strtIx:endIx,:] = lidarData['Aerosol_Extinction_Coefficient_mask'].data

        
    # Move our starting index for the output array to immediately
    #    after the end of the current data slice
    strtIx = endIx
    
    fileCount += 1


# **** Write concatenated data out to NetCDF ****
print('\n{}    Concatenation complete. Beginning netCDF write...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))
# Create a netCDF file to hold our output
rootGrp = Dataset(outFileName,'w',format='NETCDF4')
rootGrp.set_fill_on()

# Define the netCDF dimensions
gateAlt1d = rootGrp.createDimension('gateAlt1d',totalAlt)
time1d = rootGrp.createDimension('time1d',totalTime)

# Create out output variable instances within the file and
#    define metadata for each as needed
TIME = rootGrp.createVariable('time1d','f8',('time1d',),fill_value=np.nan)
TIME.long_name = 'POSIX Timestamp'
TIME.units = 'Seconds since 1 Jan 1970'
TIME.coordinates = 'time1d'

GALT = rootGrp.createVariable('gateAlt1d','f4',('gateAlt1d',),fill_value=np.nan)
GALT.long_name = 'Altitude of each gate center above MSL'
GALT.units = 'm'
GALT.coordinates = 'gateAlt1d'

TIME2D = rootGrp.createVariable('time2d','f8',('time1d','gateAlt1d'),fill_value=np.nan)
TIME2D.long_name = 'POSIX Timestamp for every gate/time'
TIME2D.units = 'Seconds since 1 Jan 1970'
TIME2D.coordinates = 'time1d gateAlt1d'

GALT2D = rootGrp.createVariable('gateAlt2d','f4',('time1d','gateAlt1d'),fill_value=np.nan)
GALT2D.long_name = 'Altitude of each gate center above MSL, for each time step (tiled)'
GALT2D.units = 'm'
GALT2D.coordinates = 'time1d gateAlt1d'

TELDIR = rootGrp.createVariable('TelescopeDir','i1',('time1d'))
TELDIR.description = 'Pointing direction of lidar telescope'
TELDIR.units = '1 = upward, 0 = downward'
TELDIR.coordinates = 'time1d'

PLNALT = rootGrp.createVariable('planeAlt','f4',('time1d'))
PLNALT.description = 'Altitude MSL of aircraft'
PLNALT.units = 'm'
PLNALT.coordinates = 'time1d'


###
AERO_BC = rootGrp.createVariable('Aerosol_Backscatter_Coefficient','f8',('time1d','gateAlt1d'),fill_value=np.nan)
AERO_BC.units = 'm-1 sr-1'
AERO_BC.description = 'Calibrated Measurement of Aerosol Backscatter Coefficient in m-1 sr-1'
AERO_BC.coordinates = 'time1d gateAlt1d'

AERO_BC_V = rootGrp.createVariable('Aerosol_Backscatter_Coefficient_variance','f4',('time1d','gateAlt1d'),fill_value=np.nan)
AERO_BC_V.coordinates = 'time1d gateAlt1d'

AERO_BC_M = rootGrp.createVariable('Aerosol_Backscatter_Coefficient_mask','i1',('time1d','gateAlt1d'))
AERO_BC_M.units = '1 = Masked, 0 = Not Masked'
AERO_BC_M.coordinates = 'time1d gateAlt1d'


###
PART_DEPOL = rootGrp.createVariable('Particle_Depolarization','f8',('time1d','gateAlt1d'),fill_value=np.nan)
PART_DEPOL.units = 'unitless'
PART_DEPOL.description = ('Propensity of Particles to depolarize (d).  This is not identical to the depolarization ratio.' 
                          'See Gimmestad: 10.1364/AO.47.003795 or Hayman and Thayer: 10.1364/JOSAA.29.000400')
PART_DEPOL.coordinates = 'time1d gateAlt1d'

PART_DEPOL_V = rootGrp.createVariable('Particle_Depolarization_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
PART_DEPOL_V.coordinates = 'time1d gateAlt1d'

PART_DEPOL_M = rootGrp.createVariable('Particle_Depolarization_mask','i1',('time1d','gateAlt1d'))
PART_DEPOL_M.units = '1 = Masked, 0 = Not Masked'
PART_DEPOL_M.coordinates = 'time1d gateAlt1d'


###
VOL_DEPOL = rootGrp.createVariable('Volume_Depolarization','f8',('time1d','gateAlt1d'),fill_value=np.nan)
VOL_DEPOL.units = 'unitless'
VOL_DEPOL.description = ('Propensity of Volume to depolarize (d).  This is not identical to the depolarization ratio.' 
                          'See Gimmestad: 10.1364/AO.47.003795 or Hayman and Thayer: 10.1364/JOSAA.29.000400')
VOL_DEPOL.coordinates = 'time1d gateAlt1d'

VOL_DEPOL_V = rootGrp.createVariable('Volume_Depolarization_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
VOL_DEPOL_V.coordinates = 'time1d gateAlt1d'

VOL_DEPOL_M = rootGrp.createVariable('Volume_Depolarization_mask','i1',('time1d','gateAlt1d'))
VOL_DEPOL_M.units = '1 = Masked, 0 = Not Masked'
VOL_DEPOL_M.coordinates = 'time1d gateAlt1d'


###
BS_RATIO = rootGrp.createVariable('Backscatter_Ratio','f8',('time1d','gateAlt1d'),fill_value=np.nan)
BS_RATIO.units = 'unitless'
BS_RATIO.description = 'Ratio of combined to molecular backscatter'
BS_RATIO.coordinates = 'time1d gateAlt1d'

BS_RATIO_V = rootGrp.createVariable('Backscatter_Ratio_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
BS_RATIO_V.coordinates = 'time1d gateAlt1d'

BS_RATIO_M = rootGrp.createVariable('Backscatter_Ratio_mask','i1',('time1d','gateAlt1d'))
BS_RATIO_M.units = '1 = Masked, 0 = Not Masked'
BS_RATIO_M.coordinates = 'time1d gateAlt1d'


###
MOLECBS_COEF = rootGrp.createVariable('Molecular_Backscatter_Coefficient','f8',('time1d','gateAlt1d'),fill_value=np.nan)
MOLECBS_COEF.units = 'm-1 sr-1'
MOLECBS_COEF.description = 'Ideal Atmosphere Molecular Backscatter Coefficient in m-1 sr-1'
MOLECBS_COEF.coordinates = 'time1d gateAlt1d'

MOLECBS_COEF_V = rootGrp.createVariable('Molecular_Backscatter_Coefficient_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
MOLECBS_COEF_V.coordinates = 'time1d gateAlt1d'


###
LOWGAIN_TBC = rootGrp.createVariable('Low_Gain_Total_Backscatter_Channel','f8',('time1d','gateAlt1d'),fill_value=np.nan)
LOWGAIN_TBC.units = 'Photon Counts'
LOWGAIN_TBC.description = 'Parallel Polarization, Low Gain, Combined Aerosol and Molecular Returns'
LOWGAIN_TBC.coordinates = 'time1d gateAlt1d'

LOWGAIN_TBC_V = rootGrp.createVariable('Low_Gain_Total_Backscatter_Channel_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
LOWGAIN_TBC_V.coordinates = 'time1d gateAlt1d'


###
HIGAIN_TBC = rootGrp.createVariable('High_Gain_Total_Backscatter_Channel','f8',('time1d','gateAlt1d'),fill_value=np.nan)
HIGAIN_TBC.units = 'Photon Counts'
HIGAIN_TBC.description = 'Parallel Polarization, High Gain, Combined Aerosol and Molecular Returns'
HIGAIN_TBC.coordinates = 'time1d gateAlt1d'

HIGAIN_TBC_V = rootGrp.createVariable('High_Gain_Total_Backscatter_Channel_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
HIGAIN_TBC_V.coordinates = 'time1d gateAlt1d'


###
MOLEC_BSC = rootGrp.createVariable('Molecular_Backscatter_Channel','f8',('time1d','gateAlt1d'),fill_value=np.nan)
MOLEC_BSC.units = 'Photon Counts'
MOLEC_BSC.description = 'Parallel Polarization\nMolecular Backscatter Returns'
MOLEC_BSC.coordinates = 'time1d gateAlt1d'

MOLEC_BSC_V = rootGrp.createVariable('Molecular_Backscatter_Channel_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
MOLEC_BSC_V.coordinates = 'time1d gateAlt1d'


###
CRSPOL = rootGrp.createVariable('Cross_Polarization_Channel','f8',('time1d','gateAlt1d'),fill_value=np.nan)
CRSPOL.units = 'Photon Counts'
CRSPOL.description = 'Cross Polarization\nCombined Aerosol and Molecular Returns'
CRSPOL.coordinates = 'time1d gateAlt1d'

CRSPOL_V = rootGrp.createVariable('Cross_Polarization_Channel_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
CRSPOL_V.coordinates = 'time1d gateAlt1d'


###
MRG_CMBND = rootGrp.createVariable('Merged_Combined_Channel','f8',('time1d','gateAlt1d'),fill_value=np.nan)
MRG_CMBND.units = 'Photon Counts'
MRG_CMBND.description = 'Merged hi/lo gain combined channel'
MRG_CMBND.coordinates = 'time1d gateAlt1d'

MRG_CMBND_V = rootGrp.createVariable('Merged_Combined_Channel_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
MRG_CMBND_V.coordinates = 'time1d gateAlt1d'

MRG_CMBND_M = rootGrp.createVariable('Merged_Combined_Channel_mask','i1',('time1d','gateAlt1d'))
MRG_CMBND_M.units = '1 = Masked, 0 = Not Masked'
MRG_CMBND_M.coordinates = 'time1d gateAlt1d'


###
AERO_EXT_COEF = rootGrp.createVariable('Aerosol_Extinction_Coefficient','f8',('time1d','gateAlt1d'),fill_value=np.nan)
AERO_EXT_COEF.units = 'm-1'
AERO_EXT_COEF.description = 'Aerosol Extinction Coefficient'
AERO_EXT_COEF.coordinates = 'time1d gateAlt1d'

AERO_EXT_COEF_V = rootGrp.createVariable('Aerosol_Extinction_Coefficient_variance','f8',('time1d','gateAlt1d'),fill_value=np.nan)
AERO_EXT_COEF_V.coordinates = 'time1d gateAlt1d'

AERO_EXT_COEF_M = rootGrp.createVariable('Aerosol_Extinction_Coefficient_mask','i1',('time1d','gateAlt1d'))
AERO_EXT_COEF_M.units = '1 = Masked, 0 = Not Masked'
AERO_EXT_COEF_M.coordinates = 'time1d gateAlt1d'




# Define global attributes
rootGrp.description = 'Concatenated High Spectral Resolution Lidar preliminary_data data'
rootGrp.flight = flight
rootGrp.history = 'Created ' + time.asctime(time.gmtime()) + ' UTC'
rootGrp.firstFile = files[lFileIx[0]]
rootGrp.lastFile = files[lFileIx[-1]]
rootGrp.lidarWavelength = 5.32e-07
rootGrp.ProcessingStatus= ('Raw Data,Removed specified times,Applied Pointwise Mask,'
                           'Nonlinear CountRate Correction for dead time 28.4 ns,Background Subtracted over [28952.46, 29694.44] m,' 
                           'Grab Range Slice 0.0 - 13874.6 m,Converted range to altitude data,'
                           'Applied Pointwise Mask,Time Resampled to dt= 2.0 s,Performed piecewise multiplication,'
                           'Profile Rescaled by array betwen 0.687502 and 0.687502,'
                           'Copy of previous profile: Molecular Backscatter Channel,'
                           'Multiplied by Molecular Backscatter Coefficient,Applied Pointwise Mask,Applied Pointwise Mask')

# Write data into netCDF variables
TIME[:] = time1d_all
GALT[:] = gateAlt_1d
TIME2D[:] = time2d_all
GALT2D[:] = gateAlt2d_all
TELDIR[:] = telescopeDir_all
PLNALT[:] = planeAlt_all

AERO_BC[:] = AerosolBackscatterCoefficient_all
AERO_BC_V[:] = AerosolBackscatterCoefficientVariance_all
AERO_BC_M[:] = AerosolBackscatterCoefficientMask_all
PART_DEPOL[:] = ParticleDepolarization_all
PART_DEPOL_V[:] = ParticleDepolarizationVariance_all
PART_DEPOL_M[:] = ParticleDepolarizationMask_all
VOL_DEPOL[:] = VolumeDepolarization_all
VOL_DEPOL_V[:] = VolumeDepolarizationVariance_all
VOL_DEPOL_M[:] = VolumeDepolarizationMask_all
BS_RATIO[:] = BackscatterRatio_all
BS_RATIO_V[:] = BackscatterRatioVariance_all
BS_RATIO_M[:] = BackscatterRatioMask_all
MOLECBS_COEF[:] = MolecularBackscatterCoefficient_all
MOLECBS_COEF_V[:] = MolecularBackscatterCoefficientVariance_all
LOWGAIN_TBC[:] = LowGainTotalBackscatterChannel_all
LOWGAIN_TBC_V[:] = LowGainTotalBackscatterChannelVariance_all
HIGAIN_TBC[:] = HighGainTotalBackscatterChannel_all
HIGAIN_TBC_V[:] = HighGainTotalBackscatterChannelVariance_all
MOLEC_BSC[:] = MolecularBackscatterChannel_all
MOLEC_BSC_V[:] = MolecularBackscatterChannelVariance_all
CRSPOL[:] = CrossPolarizationChannel_all
CRSPOL_V[:] = CrossPolarizationChannelVariance_all
MRG_CMBND[:] = MergedCombinedChannel_all
MRG_CMBND_V[:] = MergedCombinedChannelVariance_all
MRG_CMBND_M[:] = MergedCombinedChannelMask_all
AERO_EXT_COEF[:] = AerosolExtinctionCoefficient_all
AERO_EXT_COEF_V[:] = AerosolExtinctionCoefficientVariance_all
AERO_EXT_COEF_M[:] = AerosolExtinctionCoefficientMask_all



# Close the output file
rootGrp.close()

print('\nTotal script run time: {}'.format(dt.now() - scriptStrtT))

