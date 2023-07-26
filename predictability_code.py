# functions for calculating the real-time OMI of OLR model data
# based of ROMI from Kiladis, 2014 with similar filtering to Kikuchi 2012
# .90 correlation to OMI

from pathlib import Path
import os.path
import inspect

import numpy as np
import pandas as pd
import xarray as xr

import mjoanalyses.general_mjo_tools as tools
import mjoindices.tools as mjotools
import mjoindices.omi.omi_calculator as omi
import mjoindices.principal_components as pc
import mjoindices.empirical_orthogonal_functions as eof
import MJO_indices.mjo_input_tools as input_tools


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


def generate_dates_restart(raw_olr):
    """
    Generate array of valid dates from OLR dataset. Needed because model dates are weird and pandas hates them. 
    """

    comp = tools.split_time_into_components_xr(raw_olr)
    years = comp.year.values
    mons = comp.month.values
    days = comp.day.values
    
    times = [np.datetime64(f'{years[i]:04d}' + '-' + f'{mons[i]:02d}' + '-' + f'{days[i]:02d}') for i in range(len(comp))]
    
    return np.array(times, dtype=np.datetime64)


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
    pcs['Phase'] = [tools.calculate_phase(pc1,pc2) for pc1,pc2 in zip(pcs.PC1,pcs.PC2)]

    pcs['time'] = olrdata.time
    pcs = pcs.set_index('time')

    columns=['PC1', 'PC2', 'Amplitude', 'Phase']

    for col in columns:
        olrdata[col] = pcs[col].to_xarray()

    return olrdata


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
    interp_olr = tools.interpolate_spacial_grid_xr(filt2, eofdata.lat, eofdata.long)

    # projects filtered OLR onto EOFs
    olr_dates = generate_dates_restart(interp_olr)
    pcs_obj = calculate_realtime_pcs(interp_olr, eofdata, olr_dates, norm_factor)

    # calculates amplitude and phase information, converts to dataframe 
    return sort_pcs_pd(pcs_obj, interp_olr)


def iterate_romi(restart_dates, ndates, omi_eofs, daily_aves):
    """
    Envelope function for calculating the ROMI for a list of restart dates.

    :param restart_dates: list of dates for calculating ROMI.
    :param ndates: index corresponding to restart dates 
    :param omi_eofs: EOFs used for projecting the ROMI PCs.
    :param daily_aves: climatology dataset used for finding the OLR anomaly

    :returns: list of ROMI values for each restart date, for forecast and corresponding control run.
    """
    
    control_romi = []
    restart_romi = []
    
    for d in ndates:
        date = restart_dates[d]
        print(date)
        
        restart_path = Path(os.path.abspath('')).parents[0] / 'cesm_output' / f'twin_restart_{date}' / 'atm' / 'hist' 
        restart_olr = tools.load_data_xr(restart_path / f'twin_restart_{date}.FLUT.nc', 'FLUT', decode_times=True)
        
        control_run, twin_run = romi.concat_full_runs(control_olr, restart_olr, date)
        
        control_romi.append(romi.process_romi_data(control_run, omi_eofs, daily_aves))
        restart_romi.append(romi.process_romi_data(twin_run, omi_eofs, daily_aves))
        
    return control_romi, restart_romi


#######################################
# below is more for analyzing ROMI and calculating prediction skill

def iterate_restart_dates(output_path):
    """
    Find dates corresponding to restart cases in cesm_output folder.

    :param output_path: Path object where restart cases are stored
    
    :returns: list of strings of dates, sorted by time, in form "YYYY-MM-DD"
    """
    
    dates = [x.parts[-1][-10:] for x in output_path.iterdir() if x.is_dir() and "twin_restart" in x.parts[-1]]
    dates.sort()
    return dates


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


def split_restarts_by_phase(restart_dates, output_path, threshold=1.14, threshold_window=0):
    """
    Separate restart dates by phase and activity. Can adjust thresholding for active MJOs.

    :param restart_dates: total list of restarts to sort
    :param output_path: Path object to restart directory
    :param threshold: integer of ROMI amplitude to use for active MJO events. Default = 1.
    :param threshold_window: number of days before and and after restart date to use for determining
    whether an MJO event is active or not. Default is 2 days, meaning we use a 5-day window with an 
    average amplitude above :param threshold: to classify the event as active. 

    :returns: list of lists for all events, sorted into 8 phases, and active events, sorted into 8 phases. 
    Length of list = 8, where each element corresponds to the dates in that phase. 
    """

    sorted_phase = [ [] for _ in range(8) ]
    sorted_act = [ [] for _ in range(8) ]

    for idx,r in enumerate(restart_dates):
        control = xr.open_dataarray(output_path / f'twin_restart_{r}/atm/hist/romi_control_{r}.nc')

        idx_restart = 6 # index of forecast start for the way I save ROMI.

        # find phase at restart date
        phase_n = int(control[idx_restart].Phase.values) - 1
        sorted_phase[phase_n].append(r)
        # determine if restart date is during an active MJO
        if np.mean(control[idx_restart-threshold_window:idx_restart+threshold_window+1].Amplitude) > threshold:
            sorted_act[phase_n].append(r)

    return sorted_phase, sorted_act


def combine_all_romi_data(restart_dates, output_path, lenrun=67):
    """
    Separate each target date by phase and activity, based on control run. 
    Stick all information for control and restart into one array, aligned by lead/lag day
    """

    #leadlag = lenrun 
    leadlag = np.arange(-6,61)
    nruns = len(restart_dates)

    romi1c = np.empty([lenrun,nruns])
    romi2c = np.empty([lenrun,nruns])
    romi1f = np.empty([lenrun,nruns])
    romi2f = np.empty([lenrun,nruns])
    phasec = np.empty([lenrun,nruns])
    ampc = np.empty([lenrun,nruns])
    restart_date = np.empty([nruns])

    for idx,r in enumerate(restart_dates):
        control = xr.open_dataarray(output_path / f'twin_restart_{r}/atm/hist/romi_control_{r}.nc')
        restart = xr.open_dataarray(output_path / f'twin_restart_{r}/atm/hist/romi_twin_{r}.nc')

        idx_restart = 0 # index of forecast start for the way I save ROMI.

        # add information to each array
        romi1c[:,idx] = control[idx_restart:].PC1
        romi2c[:,idx] = control[idx_restart:].PC2
        romi1f[:,idx] = restart[idx_restart:].PC1
        romi2f[:,idx] = restart[idx_restart:].PC2
        phasec[:,idx] = control[idx_restart:].Phase
        ampc[:,idx] = control[idx_restart:].Amplitude


    return xr.Dataset(data_vars=dict(
                            ROMI1C=(['leadlag', 'nruns'],romi1c),
                            ROMI2C=(['leadlag', 'nruns'],romi2c),
                            ROMI1F=(['leadlag', 'nruns'],romi1f),
                            ROMI2F=(['leadlag', 'nruns'],romi2f),
                            Phase=(['leadlag', 'nruns'],phasec),
                            Amplitude=(['leadlag', 'nruns'],ampc),
                                    ),
                      coords=dict(
                            leadlag=leadlag,
                            nruns=range(nruns),
                            restart_date=("nruns",restart_dates)
                      ))


def calculate_bicor_rmse_target(restart_dates, pcs_omi, output_path, threshold=1.14):

    """
    Calculate bivariate ACC and RMSE for all forecast runs, as well as by phase and MJO strength,
    based on the given threshold. Separates by target phase and strength. 

    :param restart_dates: list of restart dates to use for calculating ACC/RMSE
    :param pcs_omi: pandas dataframe of pre-calculated OMI values for the control run, used to 
    separate target dates by phase / MJO strength.
    :param output_path: string where restart and control runs are located
    :param threshold: OMI amplitude threshold for separating target dates into strong and weak.

    :returns: xarray of ACC/RMSE for each lag day, split into all events, strong/weak targets, and
    each target phase (also by strong/weak/all events).
    """

    omi_idx = pcs_omi.set_index('Date')

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

    for r in restart_dates:
        control = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_control_{r}.nc')
        restart = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_twin_{r}.nc')

        # get OMI info for restart_date: 
        omi_slice = omi_idx[deleteLeadingZeros(r):deleteLeadingZeros(control.time.values[-1].strftime(format="%Y-%m-%d"))][['Phase','Amplitude']]

        for idx in range(len(lag)):
            # for calculating ACC
            d = idx+6 # for aligning to day 0
            num = control[d].PC1*restart[d].PC1 + control[d].PC2*restart[d].PC2
            den1 = control[d].PC1**2 + control[d].PC2**2
            den2 = restart[d].PC1**2 + restart[d].PC2**2
            
            # MSE
            mse = (control[d].PC1 - restart[d].PC1)**2 + (control[d].PC2 - restart[d].PC2)**2

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


def calculate_bicor_rmse(restart_dates, output_path = '/n/home04/sweidman/cesm_output/'):
    """
    Calculate the bivariate correlation (ACC) and RMSE across forecast runs. Takes a long time
    since needs to load pre-calculated ROMI for each forecast and sums together. 

    :param restart_dates: range of restarts used for calculating ACC and RMSE
    :param output_path: path to restart directory, string.

    :returns: np.array of ACC and RMSE by day after forecast initiation. Array of size [forecast length,2].
    First column is for ACC, second column is RMSE. 
    """
    
    temp = xr.open_dataarray(output_path + f'twin_restart_{restart_dates[0]}/atm/hist/romi_control_{restart_dates[0]}.nc')
    len_run = len(temp.time)
    n_forecasts = len(restart_dates)
    print('N forecasts: ', n_forecasts)
    
    error_by_day = np.empty([len_run,2])
    
    for d in range(len_run):
        
        num = 0 # numerator of ACC(t)
        den1 = 0 # denominator of PC1 of ACC(t)
        den2 = 0 # denominator of PC2 of ACC(t)
        
        mse = 0 # cumulative sum of mse
        
        # sums over each forecast at lag d from forecast start. 
        for r in restart_dates:
            
            # reload in each control and restart run
            control = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_control_{r}.nc')
            restart = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_twin_{r}.nc')

            # for calculating ACC
            num += control[d].PC1*restart[d].PC1 + control[d].PC2*restart[d].PC2
            den1 += control[d].PC1**2 + control[d].PC2**2
            den2 += restart[d].PC1**2 + restart[d].PC2**2
            
            # MSE
            mse += (control[d].PC1 - restart[d].PC1)**2 + (control[d].PC2 - restart[d].PC2)**2
            
        error_by_day[d,0] = num/(np.sqrt(den1)*np.sqrt(den2))
        error_by_day[d,1] = np.sqrt(mse/n_forecasts) 

    # Print where MJO is considered predictable by ACC > 0.5
    print("Correlation < 0.5 at day:", np.where(error_by_day[:,0] < .5)[0][0] - 6)
        
    return np.array(error_by_day)


def calculate_extended_romi(full_control, restart_date, omi_eofs, long_freq):
    """
    Recalculate ROMI for longer period for the signal to noise ratio. 

    :param full_control: OLR data for full control simulation
    :param restart_date: date of forecast start
    :param omi_eofs: EOFdata used for projecting OLR to find PCs
    :param long_freq: dataset with climatology / seasonal cycle for finding anomaly. 

    :returns: ROMI for control run corresponding to restart date with longer previous window. 
    """
    
    # find required start and end of simulation based on restart date
    control_start = xr.cftime_range(end=restart_date, 
                                    freq='1D', 
                                    periods=68, 
                                    calendar='noleap')[0]
    
    control_end_time = xr.cftime_range(start=restart_date, 
                                    freq='1D', 
                                    periods=86, 
                                    calendar='noleap')[-1]
        
    control_slice = full_control.sel(time = slice(control_start,control_end_time))
    
    # recalculate ROMI
    new_romi = process_romi_data(control_slice, omi_eofs, long_freq)
    
    return new_romi

def signal_noise_ratio(restart_dates, control_olr, omi_eofs, long_freq, L=25, output_path='/n/home04/sweidman/cesm_output/'):
    """
    Calculate signal and MSE for each forecast. Required recalculating ROMI for the control
    simulation since signal requires a longer prior window. Signal is calculated for the control
    simulation but should not be significantly different from the forecasts. 

    :param restart_dates: list of restart dates for calculating SNR
    :param control_olr: OLR data for full control simulation
    :param omi_eofs: EOFdata used for projecting OLR to find PCs 
    :param long_freq: dataset with climatology / seasonal cycle for finding anomaly
    :param L: window used for calculating signal. Using 51-day window because of previous literature. 
    If this changes, will need to change hardcoded indexing below. 

    :returns: np arrays of MSE and signal for each day of each forecast. Arrays of size
    [length of forecast, number of forecasts]
    """
    
    temp = xr.open_dataarray(output_path + f'twin_restart_{restart_dates[0]}/atm/hist/romi_control_{restart_dates[0]}.nc')
    len_run = len(temp.time)
    n_forecasts = len(restart_dates)
    print('N forecasts: ', n_forecasts)
    count = 0
    
    mse = np.zeros([len_run, n_forecasts])
    signal = np.zeros([len_run, n_forecasts])
    
    for r in restart_dates:
        
        print(r)
    
        control = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_control_{r}.nc')
        restart = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_twin_{r}.nc')

        # Change here if you want to use observed EOFs for calculating ROMI.  
        #control = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_obs_control_{r}.nc')
        #restart = xr.open_dataarray(output_path + f'twin_restart_{r}/atm/hist/romi_obs_twin_{r}.nc') 

        # calculate ROMI with longer prior window
        new_romi = calculate_extended_romi(control_olr, r, omi_eofs, long_freq)
        
        for d in range(len_run):
            
            mse[d, count] = (control[d].PC1 - restart[d].PC1)**2 + (control[d].PC2 - restart[d].PC2)**2
            signal[d, count] += sum([new_romi[i+18].PC1**2 + new_romi[i+18].PC2**2 for i in range(d-L, d+L)])/(2*L+1)
            
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

def calculate_ci_acc(data,n,df):

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

def calculate_ci_rmse(data,n,df):

    c1,c2 = stats.chi2.ppf([0.025,1-0.025],n)

    lb = np.sqrt(n/c1)*data
    ub = np.sqrt(n/c2)*data

    return lb, ub
