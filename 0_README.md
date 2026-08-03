Current
+ Restructure and cleanup

Next Goals
+ Seasonal Cycle code
    + applied to soil moisture
    + applied to snow depth
    + applied to soil temperature
+ extend soil temp plots down to 15m with obs on top
+ move obs in front of simulation data
+ add more variables based on data availability (precip?)
+ add era5 soil moisture plots
+ add bias corrected era5 soil moisture
+ figure out what plant data to plot
+ write metrics code
+ look at forcing data, maybe plot some of it??
+ investigate that 2010-2011 jump



Late Game
+ should this whole process be in OnDemand ??
+ check canswe for more snow depth related data
+ clean up ERA5 api calls and processing code
+ need to run CTSM for 2015-2025 to match against IQ and BC data
+ automate the interpolation based on file name ?
+ I'm having consistent issues with the time functions not matching or being weird. I should look into a standardized way to convert things - likely I need to be working in just xarray rather than the xarray pandas hybrid I've been doing
+ Some GUI front end for CTSM analysis would be cool. Possibly look into that defunct one. just having a way to do instant seasonal, monthly averages, time series, soil profiles, etc. would be a cool end product




Iqaluit data note from paper: 
"In addition to the radiation fluxes, two Campbell Scientific SR50ATH snow depth sensors and a CS655 soil water content reflectometer with a soil temperature sensor were also installed. They provide observations of snow depth, soil moisture, and soil temperature below the flux sensor suite to further help characterize the site’s radiative budget. Two flat calibration target pads were installed under each SR50ATH to ensure snow depth measurements were calibrated and recorded on a standardized surface. Finally, a Rosemount icing detector provides an indication of icing conditions (i.e., the presence of super-cooled water and an estimate of its quantity). It consists of a piezoelectric sensor that detects changes in its natural vibration frequency due to ice buildup. As such, it is useful for determining whether ice and/or frost formed on/near the surface"

CYFB-DAQ-FLUX Campbell Scientific radiation
flux sensor suite and snow/soil
depth measurements
Raw: upward and downward shortwave (pyranometer) and up-
/down/N/E/S/W longwave (pyrgeometers) radiation flux sen-
sors, SR50 snow depth and soil observations


# Workflow

1. merging files
take full folders of daily data and use a merge to get them all in one .nc file. 
```Bash
ncrcat test_folder/*.nc merged_test.nc
```
2. sanity check (1 month)
I ran a test of a month's data in both the orginal files and the merged data. Anymore than a month is too much. I also ran an equality check on two data frames
```Python
variable_df_original.equals(variable_df_merged)
```
and that came out true. Also checking time steps in the full merged files
```Bash
ncdump -v time merged_test.nc
```
3. sanity check (full time)
run the code to check the amount of days and try plotting some variable over the full time. 
4. Start comparisons
Now that the data is easy to work with, we can start observational comparisons
