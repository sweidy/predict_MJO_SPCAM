# functions for calculating the real-time OMI of OLR model data
# based of ROMI from Kiladis, 2014 with similar filtering to Kikuchi 2012

# Requires the mjoindices package from https://github.com/cghoffmann/mjoindices; some of the functions
# to calculate the ROMI are modified from the functions for OMI in the package. A companion paper to the
# package can be found at:
# Hoffmann, C.G., Kiladis, G.N., Gehne, M. and von Savigny, C., 2021. 
# A Python Package to Calculate the OLR-Based Index of the Madden- Julian-Oscillation (OMI) 
# in Climate Science and Weather Forecasting. Journal of Open Research Software, 9(1), p.9. 
# DOI: http://doi.org/10.5334/jors.331

from pathlib import Path
import os.path
import inspect

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

import mjoindices.tools as mjotools
import mjoindices.omi.omi_calculator as omi
import mjoindices.principal_components as pc
import mjoindices.empirical_orthogonal_functions as eof

def concat_full_runs(control, restart, restart_date, prev_window=50):
    """
    Prepend control run onto the restart run for time filtering and plotting. Slice
    control run to match time dimension of new restart.

    :param control: full OLR file for control run
    :param restart: full OLR file for single restart run
    :param restart_date: string of date corresponding to restart
    :param prev_window: integer of number of days before restart date to use in slices.

    :returns: tuple of xarrys with slice of control, restart. time dimension is prev_window + len(restart)
    """
    
    control_start = xr.cftime_range(end=restart_date, 
                                    freq='1D', 
                                    periods=prev_window, 
                                    calendar='noleap')[0]
    
    prev_buffer = control.sel(time = slice(control_start,restart_date))
    control_end = control.sel(time = slice(restart.time[0], restart.time[-1]))
    
    return xr.concat([prev_buffer, control_end], dim='time'), xr.concat([prev_buffer, restart], dim='time')


def anom_of_180d_runs(data, seasonal_cycle):
    """
    Finds anomaly of data (xarray) using seasonal cycle information (xarray). seasonal cycle
    must contain data for each day of year. 

    Does not have to be 180 days. seasonal cycle could be daily mean or n harmonics, etc. 
    """

    return data.groupby("time.dayofyear") - seasonal_cycle


def subtract_prev_window(data, window=40):
    """
    Remove previous rolling n-day mean from anomaly. Used to remove low frequency variability
    in data. Default uses 40-day mean (window) from standard ROMI calculation. 

    Returns only dates where full previous n-day mean has been removed (no tapering)
    """
    
    prev_window = data.rolling(time=window, center=False).mean().dropna("time") # prior mean
    
    # subtract prior mean from data. returns only dates with valid data. 
    return data - prev_window


def take_running_mean(data, window=9):
    """
    Computes 9-day running mean of data. At end of run, tapered to take 7-, 5-, 3-, and 1-day means 
    as available. Similar behavior if using a different (odd-numbered) window. 9-day window is 
    standard for ROMI calculation. Assumes index is updated as you gain observations throughout the run.

    Returns smoothed valid data. 
    """
    
    # take n-day mean for center of data
    full_roll = data.rolling(time=window, center=True).mean().dropna("time")
    
    # adds last days of run with tapered mean. 
    while window >1:
        window -= 2
        
        taper_roll = data[-window:].rolling(time=window, center=True).mean().dropna("time")
        full_roll = xr.concat([full_roll, taper_roll], dim='time')
        
    return full_roll


def calculate_realtime_pcs(filtered_olr_data, eofdata, olr_dates, normalization_factor=0.004):
    """
    Projects filtered OLR data onto EOFs from that day of year using similar approach to OMI. 
    For standard ROMI, use filtering in func: process_romi_data to calculate filtered_olr_data.
    Default to normalize by average standard deviation of PCs in 40-year control run (for SPCAM)

    :param eofdata: EOFs from previous calculation, using EOFdata type.
    :param olr_dates: np array of dates in filtered OLR data
    :param normalization_factor: constant value of standard deviation for normalizing PCs to 1. 

    :returns: PCdata for ROMI, first two principal components. 
    """

    raw_pcs = romi_regress_3dim_data_onto_eofs(filtered_olr_data, eofdata, olr_dates)
    pc1 = np.multiply(raw_pcs.pc1, normalization_factor)
    pc2 = np.multiply(raw_pcs.pc2, normalization_factor)
    return pc.PCData(raw_pcs.time, pc1, pc2)


def romi_regress_3dim_data_onto_eofs(data: object, eofdata: eof.EOFDataForAllDOYs, olr_dates) -> pc.PCData:
    """
    Similar to function in mjoindices package but using valid dates for SPCAM. 

    Finds time-dependent coefficients w.r.t the DOY-dependent EOF basis for time-dependent spatially resolved data.
    I.e. it finds the PCs for temporally resolved OLR data. But the function can also be used for other datasets,
    as long as those datasets have the same structure as the the class :class:`mjoindices.olr_handling.OLRData`.
    :param data: The data used to compute the coefficients. Should be an object of class
        :class:`mjoindices.olr_handling.OLRData` or of similar structure.
    :param eofdata: The DOY-dependent pairs of EOFs, as computed by, e.g., :func:`calc_eofs_from_olr`
    :return: The time-dependent PCs as :class:`mjoindices.principal_components.PCData`
    """

    pc1 = np.empty(data.time.size)
    pc2 = np.empty(data.time.size)

    for idx, val in enumerate(olr_dates):
        day = val
        olr_singleday = data[idx].values
        doy = mjotools.calc_day_of_year(day, eofdata.no_leap_years)

        np.reshape(olr_singleday, eofdata.eofdata_for_doy(doy).lat.size * eofdata.eofdata_for_doy(doy).long.size)

        (pc1_single, pc2_single) = omi.regress_vector_onto_eofs(
            np.reshape(olr_singleday, eofdata.eofdata_for_doy(doy).lat.size * eofdata.eofdata_for_doy(doy).long.size),
            eofdata.eof1vector_for_doy(doy),
            eofdata.eof2vector_for_doy(doy))
        pc1[idx] = pc1_single
        pc2[idx] = pc2_single
    return pc.PCData(data.time, pc1, pc2)


def split_time_into_components_xr(data: xr.DataArray) -> xr.DataArray:
    """
    Splits time of xr DataArray into yy, mm, dd
    
    :param data: xr DataArray with dimensions [time, space]
    
    :returns: xr DataArray with dimensions [time, space, year, month, day]
    """
    
    vtimes = data.time.values.astype('datetime64[ms]').astype('O')

    yy = [i.year for i in vtimes]
    mm = [i.month for i in vtimes]
    dd = [i.day for i in vtimes]

    # add new coordinates to DataArray
    data = data.assign_coords(year=('time',yy),
                     month=('time',mm),
                     day=('time',dd))
    
    return data


def generate_dates_restart(raw_olr):
    """
    Generate array of valid dates from OLR dataset. Needed because model dates are weird and pandas hates them. 
    """

    comp = split_time_into_components_xr(raw_olr)
    years = comp.year.values
    mons = comp.month.values
    days = comp.day.values
    
    times = [np.datetime64(f'{years[i]:04d}' + '-' + f'{mons[i]:02d}' + '-' + f'{days[i]:02d}') for i in range(len(comp))]
    
    return np.array(times, dtype=np.datetime64)


def calculate_phase(pc1: int, pc2: int) -> int:
    """
    Calculates phase of the MJO based on PC1 and PC2 of the OMI.

    :param pc1: integer of PC1 for some date
    :param pc2: integer of PC2 for same date

    :returns: integer phase number based on angle between PC1,PC2
    """
    
    if np.abs(pc1) >= np.abs(pc2):
        
        if pc1 >= 0:
            if pc2 >= 0:
                phase = 3
            else: # pc2 < 0
                phase = 2
        else: # pc1 < 0
            if pc2 >= 0:
                phase = 6
            else: # pc2 < 0
                phase = 7
                
    else: # pc2 > pc1
        
        if pc1 >= 0:
            if pc2 >= 0:
                phase = 4
            else: # pc2 < 0
                phase = 1
        else: # pc1 < 0
            if pc2 >= 0:
                phase = 5
            else: # pc2 < 0
                phase = 8
                
    return phase


def sort_pcs_pd(pcs_obj, olrdata):
    """
    Converts PCsdata and converts to pandas dataframe with phase and amplitude as calculated from
    principal components. 
    """

    pcs = pd.DataFrame({"PC1": pcs_obj.pc1, "PC2": pcs_obj.pc2})

    # calculate the amplitude for each day (RMS of PC1 and PC2)
    pcs['Amplitude'] = np.sqrt(pcs.PC1**2 + pcs.PC2**2)
    
    # calculate tangent for phase
    pcs['Theta'] = np.arctan(-pcs.PC1/pcs.PC2)
    pcs['Phase'] = [calculate_phase(pc1,pc2) for pc1,pc2 in zip(pcs.PC1,pcs.PC2)]

    pcs['time'] = olrdata.time
    pcs = pcs.set_index('time')

    columns=['PC1', 'PC2', 'Amplitude', 'Phase']

    for col in columns:
        olrdata[col] = pcs[col].to_xarray()

    return olrdata


def interpolate_spacial_grid_xr(data: xr.DataArray, target_lat: np.ndarray, target_long: np.ndarray,
                                bounds_error: bool=True) -> xr.DataArray:
    """
    Interpolates the OLR data linearly onto the given grids.
    No extrapolation will be done if bounds_error = True. Instead a :py:class:`ValueError` is raised 
    if the data does not cover the target grid.
    
    :param data: The data to interpolate
    :param target_lat: The new latitude grid.
    :param target_long: The new longitude grid.
    :param bounds_error: if True, will raise error instead of extrapolating. if False, will extrapolate at boundaries.
    Use bounds_error=False carefully. 
    :return: interpolated data in xarray form
    """
    
    no_days = data.time.size
    data_interp = np.empty((no_days, target_lat.size, target_long.size))
    
    for idx, t in enumerate(data.time.values):
        f = scipy.interpolate.interp2d(data.lon.values, data.lat.values, data.sel(time=t).values,
                                       kind='linear', bounds_error=bounds_error) 
        #temp = f(target_long, target_lat) 
        #data_interp[idx, :, :] = np.flip(temp, axis=0) # TODO: is this required?? 
        data_interp[idx, :, :] = f(target_long, target_lat)

    # put back in xarray
    return xr.DataArray(data_interp, coords={'lat': target_lat,
                                             'lon': target_long,
                                             'time': data['time'].values},
                        dims=['time','lat','lon']) 


def process_romi_data(raw_olr, eofdata, seasonal_cycle, low_window=40, smooth_window=9, norm_factor=0.004):
    """
    Main function for calculating ROMI. Filters data according to standard procedure and then
    projects filtered OLR onto OMI EOFs. 

    :param raw_olr: unfiltered OLR from short restart or control run
    :param eofdata: EOFdata used for describing MJO signal. Must have same leap year behavior as 
    raw OLR. 
    :param seasonal_cycle: xarray of seasonal cycle for each day of year and grid point. Could be 
    daily mean or n harmonics or something. 
    :param low_window: length of window for previous n day mean, used for removing low frequency signal
    :param smooth_window: length of window for smoothing data (taking running mean)
    :param norm_factor: constant value of standard deviation for normalizing PCs to 1. 

    :returns: pandas df of ROMI PC information, including amplitude and phase. Reminder: ROMI is about 2
    days offset from OMI, according to Kiladis 2014.
    """

    # calculate daily anomaly first
    anom_olr = anom_of_180d_runs(raw_olr, seasonal_cycle) 

    # remove low frequency signal and smooth data
    filt1 = subtract_prev_window(anom_olr, low_window)
    filt2 = take_running_mean(filt1, smooth_window)

    # interpolate to standard EOF grid
    interp_olr = interpolate_spacial_grid_xr(filt2, eofdata.lat, eofdata.long)

    # projects filtered OLR onto EOFs
    olr_dates = generate_dates_restart(interp_olr)
    pcs_obj = calculate_realtime_pcs(interp_olr, eofdata, olr_dates, norm_factor)

    # calculates amplitude and phase information, converts to dataframe 
    return sort_pcs_pd(pcs_obj, interp_olr)


#######################################
# below are functions used to analyze the pre-calculated ROMI for reproducing figures of the paper.
####################################### 

def deleteLeadingZeros(inputString):
    """
    Code from stack exchange, used to slice date strings to the correct length.
    """
    # traversing through the length of the string
    for k in range(len(inputString)):
        # checking whether the current character of a string is not 0
        if inputString[k] != '0':
            # getting the remaining string using slicing
            outputString= inputString[k::]
            # returning resultant string after removing leading 0s
            return outputString
    # returning 0 if there are no leading 0s found in the entire string
    return "0"


def split_restarts_by_phase_initial(ROMI_data, threshold=1.14):
    """
    Separate restart dates by phase and activity. Can adjust thresholding for active MJOs.

    :param ROMI_data: xrDataset containing ROMI data for each forecast run.
    :param threshold: integer of ROMI amplitude to use for active MJO events. Default = 1.14, the median 
    amplitude over the 20-year control run.

    :returns: list of lists for all events, sorted into 8 phases, and active events, sorted into 8 phases. 
    Length of list = 8, where each element corresponds to the dates in that phase. 
    """

    sorted_phase = [ [] for _ in range(8) ]
    sorted_act = [ [] for _ in range(8) ]

    for idx in range(len(ROMI_data.nruns)):
        
        # find phase at restart date
        phase_n = int(ROMI_data.Phase.sel(nruns=idx,leadlag=0)) - 1
        sorted_phase[phase_n].append(str(ROMI_data.restart_date.sel(nruns=idx).values))
        # determine if restart date is during an active MJO
        if np.mean(ROMI_data.Amplitude.sel(nruns=idx, leadlag=0) > threshold):
            sorted_act[phase_n].append(str(ROMI_data.restart_date.sel(nruns=idx).values))

    return sorted_phase, sorted_act


def slice_ROMI_data(ROMI_data, slice_dates):
    """
    slice the full ROMI dataset and select only data from specified dates in slice_dates
    used for doing calculations only with initial active/inactive/phase forecasts.

    :param ROMI_data: xrDataset containing ROMI data for each forecast run.
    :param slice_dates: list of forecast dates to restrict the dataset into 

    :returns: xrDataset with all the same information as ROMI_data, but with only the forecast information
    from the specified dates.
    """

    ROMI_slice = ROMI_data.sel(nruns = ROMI_data.nruns.loc[[r in slice_dates for r in ROMI_data.restart_date.values]])

    # reset the number of forecast runs to the smaller slice
    return ROMI_slice.assign_coords(nruns = np.arange(0,len(slice_dates)))


def calculate_bicor_rmse_target(ROMI_data, pcs_omi, threshold=1.14):

    """
    Calculate bivariate ACC and RMSE for all forecast runs, as well as by phase and MJO strength,
    based on the given threshold. Separates by target phase and strength. 

    :param ROMI_data: xrDataset containing ROMI data for each forecast run.
    :param pcs_omi: pandas dataframe of pre-calculated OMI values for the control run, used to 
    separate target dates by phase / MJO strength.
    :param threshold: OMI amplitude threshold for separating target dates into strong and weak.

    :returns: xarray of ACC/RMSE for each lag day, split into all events, strong/weak targets, and
    each target phase (also by strong/weak/all events).
    """

    lag = np.arange(0,-61,-1)
    phase = np.arange(0,9) # 0 index is for all phases
    amp = ['weak', 'strong', 'all']

    # make an array or an xr array with ^ as coordinates
    info_store = np.zeros([len(phase),len(amp),len(lag),5]) # [phase, strength, lag, type]
    # type: 1: numerator
    #       2: denominator1
    #       3: denominator2
    #       4: mse
    #       5: nforecasts

    #for r in ROMI_data.restart_date.values:
    for n in ROMI_data.nruns.values:

        r = ROMI_data.restart_date.values[n]
        # get OMI info for restart_date:
        omi_idx = pcs_omi.index[pcs_omi['Date'] == deleteLeadingZeros(r)].tolist()[0]
        omi_slice = pcs_omi[omi_idx:omi_idx+len(lag)][['Phase','Amplitude']].reset_index()

        for idx in range(len(lag)):
            # for calculating ACC
            num = float(ROMI_data.ROMI1C.sel(leadlag=idx,nruns=n)*ROMI_data.ROMI1F.sel(leadlag=idx,nruns=n) + ROMI_data.ROMI2C.sel(leadlag=idx,nruns=n)*ROMI_data.ROMI2F.sel(leadlag=idx,nruns=n))
            den1 = float(ROMI_data.ROMI1C.sel(leadlag=idx,nruns=n)**2 + ROMI_data.ROMI2C.sel(leadlag=idx,nruns=n)**2)
            den2 = float(ROMI_data.ROMI1F.sel(leadlag=idx,nruns=n)**2 + ROMI_data.ROMI2F.sel(leadlag=idx,nruns=n)**2)
            
            # MSE
            mse = float((ROMI_data.ROMI1C.sel(leadlag=idx,nruns=n) - ROMI_data.ROMI1F.sel(leadlag=idx,nruns=n))**2 + (ROMI_data.ROMI2C.sel(leadlag=idx,nruns=n) - ROMI_data.ROMI2F.sel(leadlag=idx,nruns=n))**2)

            # load in proper data: [0,2,:,:] includes all info
            # also add to phase, amplitude, and both indices based on OMI
            # adding +1 to number of forecasts (last index)
            info_store[0,2,idx,:] += [num, den1, den2, mse, 1]
            info_store[omi_slice.Phase[idx],2,idx,:] += [num, den1, den2, mse, 1]
            info_store[0,int(omi_slice.Amplitude[idx] > threshold),idx,:] += [num, den1, den2, mse, 1]
            info_store[omi_slice.Phase[idx],int(omi_slice.Amplitude[idx] > threshold),idx,:] += [num, den1, den2, mse, 1] 

    acc = info_store[:,:,:,0]/(np.sqrt(info_store[:,:,:,1])*np.sqrt(info_store[:,:,:,2]))
    rmse = np.sqrt(info_store[:,:,:,3]/info_store[:,:,:,4])  

    return xr.Dataset({
                        "acc": (["phase", "strength", "lag"], acc),
                        "rmse": (["phase", "strength", "lag"], rmse),
                        },
                        coords={"phase": phase,
                                "strength": amp,
                                "lag": lag})


def calculate_bicor_rmse_initial(ROMI_data):
    """
    Calculate the bivariate correlation (ACC) and RMSE across forecast runs for all dates provided. To restrict the calculations to 
    a specific set of dates (such as active or inactive dates), use :func slice_ROMI_data():

    :param ROMI_data: xrDataset containing ROMI data for each forecast run.

    :returns: np.array of ACC and RMSE by day after forecast initiation. Array of size [forecast length,2].
    First column is for ACC, second column is RMSE. 
    """
    
    len_run = ROMI_data.leadlag.values[-1]+1
    n_forecasts = len(ROMI_data.nruns)
    print('N forecasts: ', n_forecasts)
    
    error_by_day = np.empty([len_run,2])
    
    for idx in range(len_run):
        
        num = 0 # numerator of ACC(t)
        den1 = 0 # denominator of PC1 of ACC(t)
        den2 = 0 # denominator of PC2 of ACC(t)
        
        mse = 0 # cumulative sum of mse
        
        # sums over each forecast at lag d from forecast start. 
        for n in ROMI_data.nruns.values:

            # for calculating ACC
            num = float(ROMI_data.ROMI1C.sel(leadlag=idx,nruns=n)*ROMI_data.ROMI1F.sel(leadlag=idx,nruns=n) + ROMI_data.ROMI2C.sel(leadlag=idx,nruns=n)*ROMI_data.ROMI2F.sel(leadlag=idx,nruns=n))
            den1 = float(ROMI_data.ROMI1C.sel(leadlag=idx,nruns=n)**2 + ROMI_data.ROMI2C.sel(leadlag=idx,nruns=n)**2)
            den2 = float(ROMI_data.ROMI1F.sel(leadlag=idx,nruns=n)**2 + ROMI_data.ROMI2F.sel(leadlag=idx,nruns=n)**2)
            
            # MSE
            mse = float((ROMI_data.ROMI1C.sel(leadlag=idx,nruns=n) - ROMI_data.ROMI1F.sel(leadlag=idx,nruns=n))**2 + (ROMI_data.ROMI2C.sel(leadlag=idx,nruns=n) - ROMI_data.ROMI2F.sel(leadlag=idx,nruns=n))**2)
            
        error_by_day[idx,0] = num/(np.sqrt(den1)*np.sqrt(den2))
        error_by_day[idx,1] = np.sqrt(mse/n_forecasts) 

    # Print where MJO is considered predictable by ACC > 0.5
    print("Correlation < 0.5 at day:", np.where(error_by_day[:,0] < .5)[0][0])
        
    return np.array(error_by_day)


def signal_noise_ratio(ROMI_data, ROMI_data_ext, L=25):
    """
    Calculate signal and MSE for each forecast. Required recalculating ROMI for the control
    simulation since signal requires a longer prior window. Signal is calculated for the control
    simulation but should not be significantly different from the forecasts. 

    :param ROMI_data: xrDataset containing ROMI data for each forecast run. 
    :param ROMI_data_ext: xrDataset containing extended ROMI data for each control run period (with a 25-day buffer). 
    :param L: window used for calculating signal. Using 51-day window because of previous literature. 
    If this changes, will need to change hardcoded indexing below. 

    :returns: np arrays of MSE and signal for each day of each forecast. Arrays of size
    [length of forecast, number of forecasts]
    """
    
    len_run = ROMI_data.leadlag.values[-1]+1
    n_forecasts = len(ROMI_data.nruns)
    print('N forecasts: ', n_forecasts)
    count = 0
    
    mse = np.zeros([len_run, n_forecasts])
    signal = np.zeros([len_run, n_forecasts])
    
    for n in ROMI_data.nruns.values:
        for idx in range(len_run):
            
            mse[idx, count] = float((ROMI_data.ROMI1C.sel(leadlag=idx,nruns=n) - ROMI_data.ROMI1F.sel(leadlag=idx,nruns=n))**2 + (ROMI_data.ROMI2C.sel(leadlag=idx,nruns=n) - ROMI_data.ROMI2F.sel(leadlag=idx,nruns=n))**2)
            signal[idx, count] += sum([ROMI_data_ext.ROMI1C.sel(leadlag=idx,nruns=n)**2 + ROMI_data_ext.ROMI2C.sel(leadlag=idx,nruns=n)**2 for i in range(idx-L, idx+L)])/(2*L+1)
            
        count += 1
    
    return mse, signal



###############
# The following functions are used for calculating the Student's t-test used for finding the 95% confidence interval.
###############

def calculate_ci_signal(data, n):

    zstat = stats.t.ppf(1-0.025,n)
    s = np.std(data, axis=1)

    lb = data.mean(axis=1) - zstat*s/np.sqrt(n)
    ub = data.mean(axis=1) + zstat*s/np.sqrt(n) 

    return lb,ub

def calculate_ci_acc(data,n):

    zstat = stats.t.ppf(1-0.025,n)

    zr = np.log((1+data)/(1-data))/2
    lz = zr - zstat/np.sqrt(n-3)
    uz = zr + zstat/np.sqrt(n-3)

    lb = (np.exp(2*lz)-1)/(np.exp(2*lz)+1)
    ub = (np.exp(2*uz)-1)/(np.exp(2*uz)+1)

    return lb, ub

def calculate_ci_mse(data,n):

    c1,c2 = stats.chi2.ppf([0.025,1-0.025],n)

    lb = (n/c1)*data
    ub = (n/c2)*data

    return lb, ub

def calculate_ci_rmse(data,n):

    c1,c2 = stats.chi2.ppf([0.025,1-0.025],n)

    lb = np.sqrt(n/c1)*data
    ub = np.sqrt(n/c2)*data

    return lb, ub
