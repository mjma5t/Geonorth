# How should this work?

## Data

3 sources: model data, observational data, reanalysis data

### Model data (simplest)

1. download files of daily data for each site from Trillium. It's just dragging it into my desktop through MOBAXterm

2. merging files
Need to be done in Ubuntu. I can take full folders of daily data and use a merge to get them all in one .nc file. 
```Bash
ncrcat test_folder/*.nc merged_test.nc
```

3. sanity check (optional)
    a. I ran a test of a month's data in both the original files and the merged data. Anymore than a month is too much. I also ran an equality check on two data frames
    ```Python
    variable_df_original.equals(variable_df_merged)
    ```
    and that came out true. Also checking time steps in the full merged files
    ```Bash
    ncdump -v time merged_test.nc
    ```
    b. sanity check (full time)
    run the code to check the amount of days and try plotting some variable over the full time.

4. move files to windows environment in C:\Users\madel\Code\GeoNorth\CTSM_data\run_{X}

4. merged files are ready for standardized comparison in the master notebook for each site


### Observational data (most time consuming)

1. Download data from some source into the correct site file (Observational_data/XX_obs/raw/{variable})

2. Process data in whatever form its in through a custom notebook in (Observational_data/XX_obs/notebooks) into an xarray. One notebook per site to account for weird data handling. But each notebook does have all the variables handled. 

3. Save data as a .nc file using:
```python
ds.to_netcdf("processed/{variable}.nc")
```

### ERA5

1. call api for era5 data using the saved notebook in the ERA5 folder. the job might need to be split into multiple calls for size reasons. Data should be saved (ERA5/{variable}/raw)

2. Use the second notebook to process the for each site and variable by turning data into an xarray and then saving as a .nc file in (ERA5/{variable}/processed)


## Analysis

1 boilerplate master notebook with all the processing code. Ideally in the notebook, the only thing that should be changed is the list of site IDs as the rest of the process should be standardized.

There will also be a utils.py file with functions saved there to be run in the code.

A run_{x} folder will contain a usable duplicate of the boilerplate notebook for whatever run it is. The boilerplate notebook and util.py file can be updated with any new visuals added. A runX_tests.ipynb file is also good for any experimenting.

## Future proof

Once a new CTSM run is done, a new run folder can be added to the CTSM_data folder and a new analysis notebook can be set up