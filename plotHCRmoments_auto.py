# **** plotHCRmoments ****
# This script imports and plots radar moments from concatenated versions (_using `concatHCRmoments`_) of the HIAPER Cloud Radar data.
# 
# Typically this script will take ~30 min to plot 15-min 4-panel plots for ~7 hours of flight time (run on an early 2015 MacBook Pro).
# 
# Written by Dan Stechman
# University of Illinois at Urbana-Champaign

import pyart
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import xarray as xr
import datetime
import matplotlib.dates as mdates
import os
import sys
import argparse

dt = datetime.datetime

scriptStrtT = dt.now()

warnings.filterwarnings("ignore",category=RuntimeWarning)



parser = argparse.ArgumentParser(epilog="example: python plotHCRmoments.py -f RF01_20180116 RF02_20180119 RF05_20180126",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--flights", nargs='+', help="research flight(s) number(s) and date(s) in the form: RF##_YYYYMMDD", required=True)
parser.add_argument("-d", "--duration", help="time duration of plots in minutes", default=15, type=int)
parser.add_argument("-c", "--dataPath", help="path of concatenated data file location", default="/Volumes/SOCRATES_1/")
parser.add_argument("-s", "--savePath", help="parent directory where plots are saved", default="/Users/danstechman/GoogleDrive/School/Research/SOCRATES/UI_OU_SOCRATES_Group/SOCRATES/Plots/")
parser.add_argument("-e", "--fType", help="file type to save plots as", default="png")
parser.add_argument("-t", "--titleAppnd", help="string to append to plot titles", default="")
parser.add_argument("-a", "--saveAppnd", help="string to append to figure filename", default="")
parser.add_argument("--strtTovrd", help="If anything other than empty string, will override stored start time for given flight", default="")
parser.add_argument("--endTovrd", help="If anything other than empty string, will override stored end time for given flight", default="")

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


# Retrieve any overrides for start and end times
strtTovrd = args.strtTovrd
endTovrd = args.endTovrd


for flight in flights:
    fStrtT = dt.now()
    # Define flight-specific parameters, namely the period over which to produce plots
    #    # RF01 case below is commented - please read for more information
    if flight == 'RF01_20180116':
        # startT and endT should be strings of the format 'YYYYmmdd_HHMMSS'
        #    Be sure that the defined range is evenly divisible by your plot
        #    period (i.e., start and end at 00, 15, 30, or 45 min if plot period is every 15 min)
        if not strtTovrd:
            startT = '20180115_230000'
        else:
            startT = strtTovrd
            
        if not endTovrd:
            endT = '20180116_054500'
        else:
            endT = endTovrd
    
    elif flight == 'RF02_20180119':
        if not strtTovrd:
            startT = '20180119_004500'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180119_064500'
        else:
            endT = endTovrd
    
    elif flight == 'RF03_20180123':
        if not strtTovrd:
            startT = '20180122_211500'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180123_034500'
        else:
            endT = endTovrd
    
    elif flight == 'RF04_20180124':
        if not strtTovrd:
            startT = '20180123_233000'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180124_054500'
        else:
            endT = endTovrd
    
    elif flight == 'RF05_20180126':
        if not strtTovrd:
            startT = '20180125_230000'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180126_051500'
        else:
            endT = endTovrd
    
    elif flight == 'RF06_20180129':
        if not strtTovrd:
            startT = '20180128_230000'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180129_060000'
        else:
            endT = endTovrd
    
    elif flight == 'RF07_20180131':
        if not strtTovrd:
            startT = '20180131_010000'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180131_073000'
        else:
            endT = endTovrd
    
    elif flight == 'RF08_20180204':
        if not strtTovrd:
            startT = '20180203_233000'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180204_063000'
        else:
            endT = endTovrd
            
    elif flight == 'RF09_20180205':
        if not strtTovrd:
            startT = '20180204_230000'
        else:
            startT = strtTovrd
            
        if not endTovrd:    
            endT = '20180205_064500'
        else:
            endT = endTovrd
    
    else:
        sys.exit('flight not currently defined. Add flight case (startT and endT) to script or'
        ' define strtTovrd and/or endTovrd arguments and try again')



    # Define the path/name of our concatenated HCR data file
    hcrFile = dataPath + flight + '/HCR/' + flight + '_concat-HCR-moments.nc'

    # Define the path where figures should be saved and make any required directories
    figSavePath = savePath + flight + '/HCR/' + saveDir + '/'
    os.makedirs(figSavePath,exist_ok=True)



    # **** Read data ****
    # Read in the concatenated HCR variables
    print('{}    Reading concatenated HCR data file for {}...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S'),flight))
    hcrData = xr.open_dataset(hcrFile)

    time1d = np.asarray(hcrData['time1d'].data,dtype='datetime64[ns]')
    time1d_rnd = (pd.to_datetime(time1d)).round('1s').values
    time2d = np.asarray(hcrData['time2d'].data,dtype='datetime64[ns]')
    gateAlt = hcrData['gateAlt2d'].data
    elev = hcrData['elevation'].data

    planeAlt = hcrData['planeAlt'].data

    # Determine the time indices where the HCR is pointing downward
    radDwnwrd = np.where(elev < 0.0)

    dbz = hcrData['DBZ'].data
    vel = hcrData['VEL'].data
    width = hcrData['WIDTH'].data
    ldr = hcrData['LDR'].data
    ncp = hcrData['NCP'].data
    snrvc = hcrData['SNRVC'].data
    dbmhx = hcrData['DBMHX'].data

    # Adjust radial velocities so negative values are always downward
    vel[radDwnwrd,:] *= -1


    # **** Mask Variables ****
    # Mask variables to be plotted based on a number of (adjustable) criteria.
    print('{}    Masking radar moments...\n'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))
    dbz_masked_ncp = np.ma.masked_where((ncp < 0.2)|(gateAlt < 0)|np.isinf(dbz)|np.isnan(dbz),dbz)
    vel_masked_ncp = np.ma.masked_where((ncp < 0.2)|(gateAlt < 0)|np.isinf(vel)|np.isnan(vel),vel)
    width_masked_ncp = np.ma.masked_where((ncp < 0.2)|(gateAlt < 0)|np.isinf(width)|np.isnan(width),width)
    ldr_masked_dbmhx = np.ma.masked_where((dbmhx < -101.0)|(gateAlt < 0)|np.isinf(ldr)|np.isnan(ldr),ldr)


    # **** Data Index ID ****
    # Determine which data indices will provide data between the user-defined start and end times
    # Convert the start and end time strings into datetimes
    startT_dt = dt.strptime(startT,'%Y%m%d_%H%M%S')
    endT_dt = dt.strptime(endT,'%Y%m%d_%H%M%S')

    # Find indices of the time variable most closely matching startT_dt and endT_dt
    tMatchStrt = min(time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - startT_dt))
    startT_ix = np.squeeze(np.where(time1d_rnd == tMatchStrt))[0] # If multiple matches, just use the first one (earliest)
    tMatchEnd = min(time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - endT_dt))
    endT_ix = np.squeeze(np.where(time1d_rnd == tMatchEnd))[-1] # If multiple matches, just use the last one (latest)

    # Get the actual start/end datetimes (from the time variable, not the user-defined start/end)
    strtDT_rnd = pd.to_datetime(time1d_rnd[startT_ix])
    endDT_rnd = pd.to_datetime(time1d_rnd[endT_ix])


    # **** Create plots ****
    print('{}    Starting plot creation...'.format(dt.strftime(dt.now(),'%m/%d/%Y %H:%M:%S')))

    # Initialize the plotting frame start and end times
    tmpStrtDT = strtDT_rnd
    tmpEndDT = strtDT_rnd + tDelta


    # Loop through our data until the end time of a given plotting frame exceeds the
    #   end time defined by the user
    #   **The extra minute timedelta is a fudge factor needed due to a currently unidentified bug
    while tmpEndDT <= (endDT_rnd + datetime.timedelta(minutes=1)):
    
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
        tMatchStrt = min(time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - tmpStrtDT))
        tmpStIx = np.squeeze(np.where(time1d_rnd == tMatchStrt))[0]
        tMatchEnd = min(time1d_rnd, key=lambda x: abs(pd.to_datetime(x) - tmpEndDT))
        tmpEndIx = np.squeeze(np.where(time1d_rnd == tMatchEnd))[-1]

    
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
        im1 = ax1.pcolormesh(time2d[tmpStIx:tmpEndIx,:],gateAlt[tmpStIx:tmpEndIx,:]/1000,dbz_masked_ncp[tmpStIx:tmpEndIx,:],
                             vmin=-40,vmax=16,cmap=pyart.graph.cm.LangRainbow12)
        ax1.plot(time2d[tmpStIx:tmpEndIx,:],planeAlt[tmpStIx:tmpEndIx]/1000,
                 'k-',linewidth=6)
        ax1.set_ylim([0,7])
        cax1 = fig.colorbar(im1,ax=ax1,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax1.set_label('Reflectivity (dBZ)',fontsize=caxFsz)
        cax1.ax.set_yticklabels(cax1.ax.get_yticklabels(), fontsize=ctkFsz)
        ax1.tick_params(axis='both', which='major', labelsize=tkFsz)
        ax1.set_title('SOCRATES - {} UTC{}'.format(titleDTstr,titleAppnd),fontsize=ttlFsz) 
        ax1.grid()


        # Plot radial velocity and label the y-axis
        im2 = ax2.pcolormesh(time2d[tmpStIx:tmpEndIx,:],gateAlt[tmpStIx:tmpEndIx,:]/1000,vel_masked_ncp[tmpStIx:tmpEndIx,:],
                             vmin=-5,vmax=5,cmap=pyart.graph.cm.Wild25_r)
        ax2.plot(time2d[tmpStIx:tmpEndIx,:],planeAlt[tmpStIx:tmpEndIx]/1000,
                 'k-',linewidth=6)
        ax2.set_ylim([0,7])
        cax2 = fig.colorbar(im2,ax=ax2,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax2.set_label('Radial Velocity (m/s)',fontsize=caxFsz)
        cax2.ax.set_yticklabels(cax2.ax.get_yticklabels(), fontsize=ctkFsz)
        ax2.tick_params(axis='both', which='major', labelsize=tkFsz)
        ax2.set_ylabel('Altitude (km)',fontsize=axFsz)
        ax2.grid()


        # Plot LDR
        im3 = ax3.pcolormesh(time2d[tmpStIx:tmpEndIx,:],gateAlt[tmpStIx:tmpEndIx,:]/1000,ldr_masked_dbmhx[tmpStIx:tmpEndIx,:],
                             vmin=-40,vmax=0,cmap=pyart.graph.cm.NWSRef)
        ax3.plot(time2d[tmpStIx:tmpEndIx,:],planeAlt[tmpStIx:tmpEndIx]/1000,
                 'k-',linewidth=6)
        ax3.set_ylim([0,7])
        cax3 = fig.colorbar(im3,ax=ax3,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax3.ax.set_yticklabels(cax3.ax.get_yticklabels(), fontsize=ctkFsz)
        cax3.set_label('LDR (dB)',fontsize=caxFsz)
        ax3.tick_params(axis='both', which='major', labelsize=tkFsz)
        ax3.grid()


        # Plot spectral width and label the x-axis (time)
        im4 = ax4.pcolormesh(time2d[tmpStIx:tmpEndIx,:],gateAlt[tmpStIx:tmpEndIx,:]/1000,width_masked_ncp[tmpStIx:tmpEndIx,:],
                             vmin=0,vmax=3,cmap=pyart.graph.cm.RefDiff)
        ax4.plot(time2d[tmpStIx:tmpEndIx,:],planeAlt[tmpStIx:tmpEndIx]/1000,
                 'k-',linewidth=6)
        cax4 = fig.colorbar(im4,ax=ax4,fraction=0.05,shrink=caxShrnk,pad=0.008)
        cax4.set_label('Spectral Width (m/s)',fontsize=caxFsz,)
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
        saveStr = '{}HCR-moments_{}{}.{}'.format(figSavePath,saveDTstr,saveAppnd,fType)
        fig.savefig(saveStr,bbox_inches='tight')
        

        # Set the start and end times for the next plot
        tmpStrtDT = tmpEndDT
        tmpEndDT += tDelta

    hcrData.close()
    print('\n\tPlotting time for {}: {}\n'.format(flight, dt.now() - fStrtT))
    
print('Plotting complete.\n\tTotal script time: {}'.format(dt.now() - scriptStrtT))