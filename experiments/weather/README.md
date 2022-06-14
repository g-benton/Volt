This directory contains the code needed to run Volt+Magpie on wind speed data taken from the [U.S. Climate Reference Network](https://www.ncei.noaa.gov/access/crn/).


To source the data first walk through the `make_wind_dataset` notebook.

To generate forecasts for a station then run

```{bash}
python GPGenerator.py 
    --kernel={volt, sm, matern} ## kernel choice
    --stn_idx=0 ## station index in the dataset
    --mean={ewma, constant} ## mean choice
    --ntrain=400 ## training window
    --n_test_times=100 ## number of test time points
```