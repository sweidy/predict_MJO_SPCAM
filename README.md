# predict_MJO_SPCAM

Includes data and code for analyzing SPCAM potential predictability experiment forecasts. Should include all the data needed for reproducing the figures in the paper (but please email me if you find something missing!)

Contents: 

    - predictability_code.py: Includes functions used to calculate the ROMI, then functions used to perform the analyses 
    that were used to create the figures in the paper.
    - ROMI_each_forecast.nc: file containing the ROMI information (PC1 and PC2) for each forecast run and the ROMI values for the 
    corresponding dates of the control simulation. Also includes phase and amplitude of the control ROMI for each forecast period. 
    Best if opened as an xarray.Dataset.
    - ROMI_ext_each_forecast.nc: file containing just ROMI1 and ROMI2 for the control simulation, separated by each forecast period 
    with a 25-day buffer. Used for calculating the sliding 51-day window, which requires a longer time period than the other calculations.
    - spcam_clim.nc: daily climatology (mean and first 3 harmonics of the seasonal cycle) of SPCAM from a 40-year simulation, 
    used for finding OLR climatology. Compressed using ncpdq from NCO.
    - PCs_twin_control_20yrs.txt: Principal components of the OMI (not real time) of the 20-year control run. Very similar to the
    ROMI, but does not have issues with tapering at the end of each forecast time series.

Modifications to the SPCAM source code (original found here: https://wiki.ucar.edu/pages/viewpage.action?pageId=205489281) are not included, but can be shared upon request. 

Zenodo repository here: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8190698.svg)](https://doi.org/10.5281/zenodo.8190698)


