

# =============================================================================
# region import packages


# basic library
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle

import sys  # print(sys.path)
sys.path.append(
    '/Users/gao/OneDrive - whu.edu.cn/ETH/Courses/4. Semester/DEoAI')
sys.path.append('/project/pr94/qgao/DEoAI')
sys.path.append('/scratch/snx3000/qgao')


# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
import dask
dask.config.set({"array.slicing.split_large_chunks": False})

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    months,
    years,
    years_months,
    timing,
    quantiles,
    folder_1km,
    extent1km,
    extent3d_m,
    extent3d_g,
    extent3d_t,
    extent12km,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind
)


# endregion


# =============================================================================
# region calculate strength and direction

wind_earth_1h_100m_strength_direction = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

rlon = wind_earth_1h_100m_strength_direction.rlon.data
rlat = wind_earth_1h_100m_strength_direction.rlat.data
lon = wind_earth_1h_100m_strength_direction.lon.data
lat = wind_earth_1h_100m_strength_direction.lat.data

wind_earth_1h_100m_strength_statistics = xr.Dataset(
    {"strength_quantiles": (
        ("quantiles", "rlat", "rlon"),
        np.zeros((len(quantiles[1]),
                  len(rlat),
                  len(rlon),
                  ))),
     "lat": (("rlat", "rlon"), lat),
     "lon": (("rlat", "rlon"), lon),
     },
    coords={
        "quantiles": quantiles[1],
        "rlat": rlat,
        "rlon": rlon,
    }
)

q = quantiles[0]
axis = 0

for i in range(21):
    arr = wind_earth_1h_100m_strength_direction.strength[
        :, np.arange(0, 40) + i * 40, :].values
    print('array loaded ' + str(i) + "/21 " + str(datetime.datetime.now()))
    
    wind_earth_1h_100m_strength_statistics.strength_quantiles[
        :, np.arange(0, 40) + i * 40, :
    ] = get_statistics(
        arr=arr, q=q, axis=axis
    )
    print(str(i) + '/21' + str(datetime.datetime.now()))

wind_earth_1h_100m_strength_statistics.to_netcdf(
    "scratch/wind_earth/wind_earth_1h_100m_strength_statistics_2010.nc"
)

'''
# check
ddd = xr.open_dataset(
    "scratch/wind_earth/wind_earth_1h_100m_strength_statistics_2010.nc"
)

eee = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

i = 200
print(np.array(ddd.strength_quantiles[0, i, i]))
print(np.array(np.min(eee.strength[:, i, i])))
print(np.array(ddd.strength_quantiles[8, i, i]))
print(np.array(np.max(eee.strength[:, i, i])))

print(np.array(ddd.strength_quantiles[11, i, i]))
print(np.array(ddd.strength_quantiles[6, i, i]) - \
    np.array(ddd.strength_quantiles[2, i, i]))

print(np.array(ddd.strength_quantiles[13, i, i]))
print(np.array(np.mean(eee.strength[:, i, i])))

'''

# endregion



# /project/pr94/qgao/miniconda3/envs/deoai/bin/python -c "from IPython import start_ipython; start_ipython()" --no-autoindent /project/pr94/qgao/DEoAI/DEoAI_analysis/python_codes/03_wind/3.1_rotate_wind_1km_100m.py





