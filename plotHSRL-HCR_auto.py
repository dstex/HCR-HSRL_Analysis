
# **** plotHSRL-HCR ****
# This script imports and plots HIAPER Cloud Radar reflectivity from concatenated versions (_using `concatHCRmoments`_) of 
#    the associated cfradial files, along with a number of variables from the High Spectral Resolution Lidar concatenated 
#    versions (_using `concatHSRLprelimData`_) of the associated "preliminary_data" files.
# Typically this script will take ~30 min to plot 15-min 4-panel plots for ~7 hours of flight time (run on an early 2015 MacBook Pro).
# 
# Written by Dan Stechman
# University of Illinois at Urbana-Champaign

import pyart
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
import datetime
import matplotlib.dates as mdates
import matplotlib.colors as colors
import warnings
import os
import argparse

dt = datetime.datetime

scriptStrtT = dt.now()

warnings.filterwarnings("ignore",category=RuntimeWarning)



parser = argparse.ArgumentParser(epilog="example: python plotHSRL-HCR.py -f RF01_20180116 RF02_20180119 RF05_20180126",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--flights", nargs='+', help="research flight(s) number(s) and date(s) in the form: RF##_YYYYMMDD")
parser.add_argument("-d", "--duration", help="time duration of plots in minutes", default=15, type=int)
parser.add_argument("-c", "--dataPath", help="path of concatenated data file location", default="/Volumes/SOCRATES_1/")
parser.add_argument("-s", "--savePath", help="parent directory where plots are saved", default="/Users/danstechman/GoogleDrive/School/Research/SOCRATES/UI_OU_SOCRATES_Group/SOCRATES/Plots/")
parser.add_argument("-e", "--fType", help="file type to save plots as", default="png")
parser.add_argument("-t", "--titleAppnd", help="string to append to plot titles", default="")
parser.add_argument("-a", "--saveAppnd", help="string to append to figure filename", default="")

args = parser.parse_args()



# This variable should match the name of the parent directory containing
#    the flight data for the given mission. Used in defining file input 
#    and output names
flights = args.flights

# Determine duration of plotting periods
duration = args.duration
dur = duration
tDelta = datetime.timedelta(minutes=dur)

# Specify the parent path of the data directory where input/output are stored
dataPath = args.dataPath

# Define various plot saving parameters
savePath = args.savePath
saveDir = '{}min_4panel'.format(dur)
fType = args.fType
# Optional strings to append to the title and/or filename for special cases
titleAppnd = args.titleAppnd
saveAppnd = args.saveAppnd



for flight in flights:
    fStrtT = dt.now()
    # Define flight-specific parameters, namely the period over which to produce plots
    if flight == 'RF01_20180116':
        # startT and endT should be strings of the format 'YYYYmmdd_HHMMSS'
        #    Be sure that the defined range is evenly divisible by your plot
        #    period (i.e., start and end at 00, 15, 30, or 45 min if plot period is every 15 min)
#         startT = '20180115_230000'
#         endT = '20180116_053000'
        startT = '20180116_020000'
        endT = '20180116_021500'
    
    if flight == 'RF02_20180119':
        startT = '20180119_013000'
        endT = '20180119_064500'
        
    if flight == 'RF03_20180123':
        startT = '20180122_220000'
        endT = '20180123_033000'
    
    if flight == 'RF04_20180124':
#         startT = '20180124_000000'
#         endT = '20180124_054500'
        startT = '20180124_024500'
        endT = '20180124_030000'
    
    if flight == 'RF05_20180126':
        startT = '20180125_230000'
        endT = '20180126_051500'
    
    if flight == 'RF06_20180129':
        startT = '20180128_230000'
        endT = '20180129_060000'
    
    if flight == 'RF07_20180131':
        startT = '20180131_010000'
        endT = '20180131_080000'

   
    # Create a timedelta corresponding to out plotting period
    tDelta = datetime.timedelta(minutes=dur)

    # Define the path/name of our concatenated data files
    hcrFile = dataPath + flight + '/HCR/' + flight + '_concat-HCR-moments.nc'
    hsrlFile = dataPath + flight + '/HSRL/' + flight + '_concat-HSRL-prelimData.nc'

    # Define the path where figures should be saved and make any required directories
    figSavePath = savePath + flight + '/HSRL/' + saveDir + '/'
    os.makedirs(figSavePath,exist_ok=True)


    # **** Read HCR Data ****
    # Read in the concatenated HCR variables
    print('{}    Reading concatenated HCR data file for {}...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S'),flight))
    hcrData = xr.open_dataset(hcrFile)

    hcr_time1d = np.asarray(hcrData['time1d'].data,dtype='datetime64[ns]')
    hcr_time1d_rnd = (pd.to_datetime(hcr_time1d)).round('1s').values
    hcr_time2d = np.asarray(hcrData['time2d'].data,dtype='datetime64[ns]')
    hcr_gateAlt = hcrData['gateAlt2d'].data

    hcr_dbz = hcrData['DBZ'].data
    hcr_ncp = hcrData['NCP'].data


    # **** Read HSRL Data ****
    # Read in the concatenated HSRL variables
    print('{}    Reading concatenated HSRL data file for {}...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S'),flight))
    hsrlData = xr.open_dataset(hsrlFile,decode_times=False)

    hsrl_time1d = np.asarray([dt.fromtimestamp(t) for t in hsrlData['time1d'].data])
    hsrl_time2d = np.transpose(np.tile(hsrl_time1d,(hsrlData['time2d'].data.shape[1],1)))
    hsrl_gateAlt = hsrlData['gateAlt2d'].data
    planeAlt = hsrlData['planeAlt'].data

    hsrl_aerosolBackCoef = np.ma.masked_array(hsrlData['Aerosol_Backscatter_Coefficient'].data, mask=hsrlData['Aerosol_Backscatter_Coefficient_mask'].data)
    hsrl_aerosolExtnctCoef = np.ma.masked_array(hsrlData['Aerosol_Extinction_Coefficient'].data, mask=hsrlData['Aerosol_Extinction_Coefficient_mask'].data)
    hsrl_partDepol = np.ma.masked_array(hsrlData['Particle_Depolarization'].data, mask=hsrlData['Particle_Depolarization_mask'].data)


    # **** Mask Variables ****
    # Mask variables to be plotted based on a number of (adjustable) criteria.
    print('{}    Masking radar moments...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))
    dbz_masked_ncp = np.ma.masked_where((hcr_ncp < 0.2)|(hcr_gateAlt < 0)|np.isnan(hcr_dbz)|np.isinf(hcr_dbz),hcr_dbz)


    # **** Data Index ID ****
    # Determine which data indices will provide data between the user-defined start and end times
    # Convert the start and end time strings into datetimes
    startT_dt = dt.strptime(startT,'%Y%m%d_%H%M%S')
    endT_dt = dt.strptime(endT,'%Y%m%d_%H%M%S')

    ## HCR ##
    # Find indices of the time variable most closely matching startT_dt and endT_dt
    hcr_tMatchStrt = min(hcr_time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - startT_dt))
    hcr_startT_ix = np.squeeze(np.where(hcr_time1d_rnd == hcr_tMatchStrt))[0] # If multiple matches, just use the first one (earliest)
    hcr_tMatchEnd = min(hcr_time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - endT_dt))
    hcr_endT_ix = np.squeeze(np.where(hcr_time1d_rnd == hcr_tMatchEnd))[-1] # If multiple matches, just use the last one (latest)

    # Get the actual start/end datetimes (from the time variable, not the user-defined start/end)
    hcr_strtDT_rnd = pd.to_datetime(hcr_time1d_rnd[hcr_startT_ix])
    hcr_endDT_rnd = pd.to_datetime(hcr_time1d_rnd[hcr_endT_ix])


    ## HSRL ##
    # Find indices of the time variable most closely matching startT_dt and endT_dt
    hsrl_tMatchStrt = min(hsrl_time1d, key=lambda x: abs(x - startT_dt))
    hsrl_startT_ix = np.squeeze(np.where(hsrl_time1d == hsrl_tMatchStrt))
    hsrl_tMatchEnd = min(hsrl_time1d, key=lambda x: abs(x - endT_dt))
    hsrl_endT_ix = np.squeeze(np.where(hsrl_time1d == hsrl_tMatchEnd))

    # Get the actual start/end datetimes (from the time variable, not the user-defined start/end)
    hsrl_strtDT_rnd = hsrl_time1d[hsrl_startT_ix]
    hsrl_endDT_rnd = hsrl_time1d[hsrl_endT_ix]

    # Check to see if our final start and end times match between HCR and HSRL
    if (hsrl_endDT_rnd == hcr_endDT_rnd) and (hsrl_strtDT_rnd == hcr_strtDT_rnd):
        print('HCR and HSRL start/end times match. Continuing...\n')
    else:
        print('HCR and HSRL start/end times do not match.\nCheck time alignments (if script does not fail) and consider revising'\
                ' startT and endT for this flight')
        print('startDT:\n\tHCR: {}\n\tHSRL:{}\nendDT:\n\tHCR: {}\n\tHSRL:{}\n'.format(dt.strftime(hcr_strtDT_rnd,'%Y%m%d %H:%M:%S'),
                                                                                   dt.strftime(hsrl_strtDT_rnd,'%Y%m%d %H:%M:%S'),
                                                                                   dt.strftime(hcr_endDT_rnd,'%Y%m%d %H:%M:%S'),
                                                                                   dt.strftime(hsrl_endDT_rnd,'%Y%m%d %H:%M:%S')))


    # **** Create plots ****
    print('{}    Starting plot creation...'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))

    # Initialize the plotting frame start and end times
    #    If we've made it here, the HCR and HSRL times match, so just arbitrarily use HCR times
    tmpStrtDT = hcr_strtDT_rnd
    tmpEndDT = hcr_strtDT_rnd + tDelta

    # Loop through our data until the end time of a given plotting frame exceeds the
    #   end time defined by the user
    #   **The extra minute timedelta is a fudge factor needed due to a currently unidentified bug
    while tmpEndDT <= (hcr_endDT_rnd + datetime.timedelta(minutes=1)):
    
        # Define plot title and filename strings depending on the time range of the plot,
        #    and whether the plot frame spans two days (i.e., don't print the date twice, 
        #    if it doesn't actually change)
        if tmpStrtDT.day == tmpEndDT.day:
            titleDTstr = '{} - {}-{}'.format(dt.strftime(tmpStrtDT,'%Y%m%d'),dt.strftime(tmpStrtDT,'%H:%M:%S'),
                                             dt.strftime(tmpEndDT,'%H:%M:%S'))
            if tDelta <= datetime.timedelta(minutes=5):
                saveDTstr = '{}_{}-{}'.format(dt.strftime(tmpStrtDT,'%Y%m%d'),dt.strftime(tmpStrtDT,'%H%M%S'),
                                                 dt.strftime(tmpEndDT,'%H%M%S'))
            else:
                saveDTstr = '{}_{}-{}'.format(dt.strftime(tmpStrtDT,'%Y%m%d'),dt.strftime(tmpStrtDT,'%H%M'),
                                                 dt.strftime(tmpEndDT,'%H%M'))
        else:
            titleDTstr = '{}-{} - {}-{}'.format(dt.strftime(tmpStrtDT,'%Y%m%d'),dt.strftime(tmpStrtDT,'%H:%M:%S'),
                                             dt.strftime(tmpEndDT,'%Y%m%d'),dt.strftime(tmpEndDT,'%H:%M:%S'))
            if tDelta <= datetime.timedelta(minutes=5):
                saveDTstr = '{}_{}-{}_{}'.format(dt.strftime(tmpStrtDT,'%Y%m%d'),dt.strftime(tmpStrtDT,'%H%M'),
                                                 dt.strftime(tmpEndDT,'%Y%m%d'),dt.strftime(tmpEndDT,'%H%M%S'))
            else:
                saveDTstr = '{}_{}-{}_{}'.format(dt.strftime(tmpStrtDT,'%Y%m%d'),dt.strftime(tmpStrtDT,'%H%M'),
                                                 dt.strftime(tmpEndDT,'%Y%m%d'),dt.strftime(tmpEndDT,'%H%M'))
    
        print('\tPlotting {}'.format(titleDTstr))
    
        # If our plotting period is longer than 5 minutes, don't show seconds in the xtick labels
        if tDelta <= datetime.timedelta(minutes=5):
            xtick_formatter = mdates.DateFormatter(fmt='%H:%M:%S')
        else:
            xtick_formatter = mdates.DateFormatter(fmt='%H:%M')
    
    
        # Find start and end indices most closely matching current plotting frame bounds
        hcr_tMatchStrt = min(hcr_time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - tmpStrtDT))
        hcr_tmpStIx = np.squeeze(np.where(hcr_time1d_rnd == hcr_tMatchStrt))[0]
        hcr_tMatchEnd = min(hcr_time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - tmpEndDT))
        hcr_tmpEndIx = np.squeeze(np.where(hcr_time1d_rnd == hcr_tMatchEnd))[-1]
    
        hsrl_tMatchStrt = min(hsrl_time1d, key=lambda x: abs(x - tmpStrtDT))
        hsrl_tmpStIx = np.squeeze(np.where(hsrl_time1d == hsrl_tMatchStrt))
        hsrl_tMatchEnd = min(hsrl_time1d, key=lambda x: abs(x - tmpEndDT))
        hsrl_tmpEndIx = np.squeeze(np.where(hsrl_time1d == hsrl_tMatchEnd))

    
        # Initialize figure with 4 subplots, sharing x and y axes
        # Figure size should *roughly* yield a 1:1 ratio for the plot dimensions, but this is
        #    entirely dependent on aircraft speed at any given time
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, sharey=True,figsize=(16*(dur/5.),28))
        axFsz = 32
        caxFsz = 28
        ttlFsz = 36
        tkFsz = 24
        ctkFsz = 20
        caxShrnk = 0.9

        
        # Plot reflectivity and create our figure title
        im1 = ax1.pcolormesh(hcr_time2d[hcr_tmpStIx:hcr_tmpEndIx,:],hcr_gateAlt[hcr_tmpStIx:hcr_tmpEndIx,:]/1000,dbz_masked_ncp[hcr_tmpStIx:hcr_tmpEndIx,:],
                             vmin=-40,vmax=16,cmap=pyart.graph.cm.LangRainbow12)
        ax1.plot(hsrl_time1d[hsrl_tmpStIx:hsrl_tmpEndIx],planeAlt[hsrl_tmpStIx:hsrl_tmpEndIx]/1000,
                 'k-',linewidth=6)
        ax1.set_ylim([0,7])
        cax1 = fig.colorbar(im1,ax=ax1,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax1.set_label('Reflectivity (dBZ)',fontsize=caxFsz)
        cax1.ax.set_yticklabels(cax1.ax.get_yticklabels(), fontsize=ctkFsz)
        ax1.tick_params(axis='both', which='major', labelsize=tkFsz)
        ax1.set_title('SOCRATES - {} UTC{}'.format(titleDTstr,titleAppnd),fontsize=ttlFsz) 
        ax1.grid()


        # Plot aerosol backscatter coefficient and label the y-axis
        im2 = ax2.pcolormesh(hsrl_time2d[hsrl_tmpStIx:hsrl_tmpEndIx,:],hsrl_gateAlt[hsrl_tmpStIx:hsrl_tmpEndIx,:]/1000,
                             hsrl_aerosolBackCoef[hsrl_tmpStIx:hsrl_tmpEndIx,:],
                             norm=colors.LogNorm(vmin=1e-8,vmax=1e-4),
                             cmap=plt.cm.nipy_spectral)
        ax2.plot(hsrl_time1d[hsrl_tmpStIx:hsrl_tmpEndIx],planeAlt[hsrl_tmpStIx:hsrl_tmpEndIx]/1000,
                 'k-',linewidth=6)
        ax2.set_ylim([0,7])
        cax2 = fig.colorbar(im2,ax=ax2,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax2.set_label('Aerosol Backscatter Coefficient\n($m^{-1}sr^{-1}$)',fontsize=caxFsz*0.7)
        cax2.ax.set_yticklabels(cax2.ax.get_yticklabels(), fontsize=ctkFsz)
        ax2.tick_params(axis='both', which='major', labelsize=tkFsz)
        ax2.set_ylabel('Altitude (km)',fontsize=axFsz)
        ax2.grid()


        # Plot aerosol extinction coefficient
        im3 = ax3.pcolormesh(hsrl_time2d[hsrl_tmpStIx:hsrl_tmpEndIx,:],hsrl_gateAlt[hsrl_tmpStIx:hsrl_tmpEndIx,:]/1000,
                             hsrl_aerosolExtnctCoef[hsrl_tmpStIx:hsrl_tmpEndIx,:],
                             norm=colors.LogNorm(vmin=1e-5,vmax=1e-2),
                             cmap=plt.cm.nipy_spectral)
        ax3.plot(hsrl_time1d[hsrl_tmpStIx:hsrl_tmpEndIx],planeAlt[hsrl_tmpStIx:hsrl_tmpEndIx]/1000,
                 'k-',linewidth=6)
        ax3.set_ylim([0,7])
        cax3 = fig.colorbar(im3,ax=ax3,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax3.ax.set_yticklabels(cax3.ax.get_yticklabels(), fontsize=ctkFsz)
        cax3.set_label('Aerosol Extinction Coefficient ($m^{-1}$)',fontsize=caxFsz*.7)
        ax3.tick_params(axis='both', which='major', labelsize=tkFsz)
        ax3.grid()


        # Plot particle depolarization and label the x-axis (time)
        im4 = ax4.pcolormesh(hsrl_time2d[hsrl_tmpStIx:hsrl_tmpEndIx,:],hsrl_gateAlt[hsrl_tmpStIx:hsrl_tmpEndIx,:]/1000,
                             hsrl_partDepol[hsrl_tmpStIx:hsrl_tmpEndIx,:],
                             vmin=0,vmax=1,
                             cmap=pyart.graph.cm.LangRainbow12)
        ax4.plot(hsrl_time1d[hsrl_tmpStIx:hsrl_tmpEndIx],planeAlt[hsrl_tmpStIx:hsrl_tmpEndIx]/1000,
                 'k-',linewidth=6)
        cax4 = fig.colorbar(im4,ax=ax4,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax4.set_label('Particle Depolarization',fontsize=caxFsz)
        cax4.ax.set_yticklabels(cax4.ax.get_yticklabels(), fontsize=ctkFsz)
        ax4.set_xlabel('Time (UTC)',fontsize=axFsz)
        ax4.xaxis.set_major_locator(mdates.MinuteLocator())
        ax4.xaxis.set_major_formatter(xtick_formatter)
        ax4.tick_params(axis='both', which='major', labelsize=tkFsz)
        ax4.grid()


        # Clean up the date format a bit further and remove extra whitespace between subplots
        fig.autofmt_xdate()
        fig.subplots_adjust(hspace=0.08)


        # Save the output figure
        saveStr = '{}HSRL-HCR_{}{}.{}'.format(figSavePath,saveDTstr,saveAppnd,fType)
        fig.savefig(saveStr,bbox_inches='tight')

        # Set the start and end times for the next plot
        tmpStrtDT = tmpEndDT
        tmpEndDT += tDelta
    
    hcrData.close()
    hsrlData.close()
    print('\n\tPlotting time for {}: {}\n'.format(flight, dt.now() - fStrtT))
    
print('Plotting complete.\n\tTotal script time: {}'.format(dt.now() - scriptStrtT))