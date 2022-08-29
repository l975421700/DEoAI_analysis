

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


# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

fontprop_tnr = fm.FontProperties(
    fname='/project/pr94/qgao/DEoAI/data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['backend'] = "Qt4Agg"

plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

# mpl.get_backend()

import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
)


# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate

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
# region rotate wind

print(datetime.datetime.now())

folder = '/1h_100m/'

with xr.open_dataset(folder_1km + folder + 'lffd20060101000000z.nc') as ds:
    rlon = ds.rlon.data
    rlat = ds.rlat.data
    lon = ds.lon.data
    lat = ds.lat.data
    
    pollat = ds.rotated_pole.grid_north_pole_latitude
    pollon = ds.rotated_pole.grid_north_pole_longitude
    
    pollat_sin = np.sin(np.deg2rad(pollat))
    pollat_cos = np.cos(np.deg2rad(pollat))
    
    lon_rad = np.deg2rad(pollon - lon)
    lat_rad = np.deg2rad(lat)
    
    arg1 = pollat_cos * np.sin(lon_rad)
    arg2 = pollat_sin * np.cos(lat_rad) - pollat_cos * \
        np.sin(lat_rad)*np.sin(lon_rad)
    
    norm = 1.0/np.sqrt(arg1**2 + arg2**2)


# np.arange(48, len(years_months))
for k in np.arange(35, 48):
    begin_time = datetime.datetime.now()
    print(begin_time)
    
    filelist = np.array(sorted(
        glob.glob(folder_1km + folder + 'lffd20' +
                  years_months[k] + '*[0-9z].nc')
    ))
    
    # create a file to store the results
    time = pd.date_range(
        "20" + years_months[k][0:2] + "-" + years_months[k][2:4] +
        "-01-00",
        "20" + years_months[k][0:2] + "-" + years_months[k][2:4] +
        "-" + filelist[-1][-12:-10] + "-23",
        freq="60min")
    wind_earth_1h_100m = xr.Dataset(
        {"u_earth": (
            ("time", "rlat", "rlon"),
            np.zeros((len(time), len(rlat), len(rlon)))),
         "v_earth": (
            ("time", "rlat", "rlon"),
            np.zeros((len(time), len(rlat), len(rlon)))),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon),
         },
        coords={
            "time": time,
            "rlat": rlat,
            "rlon": rlon,
        }
    )
    
    # calculate vorticity
    
    ncfiles = xr.open_mfdataset(
        filelist, concat_dim="time",
        data_vars='minimal', coords='minimal', compat='override'
    )
    
    print(datetime.datetime.now())
    
    u = ncfiles.U.data.squeeze()
    v = ncfiles.V.data.squeeze()
    
    wind_earth_1h_100m.u_earth[:, :, :] = u * arg2 * norm + v * arg1 * norm
    wind_earth_1h_100m.v_earth[:, :, :] = -u * arg1 * norm + v * arg2 * norm
    
    # write out files
    wind_earth_1h_100m.to_netcdf(
        "/project/pr94/qgao/DEoAI/scratch/wind_earth/wind_earth_" + folder[1:] + "wind_earth_" + folder[1:-1] + "20" + years_months[k] + ".nc"
    )
    del wind_earth_1h_100m
    del ncfiles
    print(str(k) + "/" + str(len(years_months)) + "   " +
          str(datetime.datetime.now() - begin_time))

'''
# check
checkfile = xr.open_dataset('/project/pr94/qgao/DEoAI/scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m201512.nc')

nc1h_100m = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_100m/lffd20151231230000z.nc')
u = nc1h_100m.U.squeeze().data
v = nc1h_100m.V.squeeze().data
lat = nc1h_100m.lat.data
lon = nc1h_100m.lon.data
pollat = nc1h_100m.rotated_pole.grid_north_pole_latitude
pollon = nc1h_100m.rotated_pole.grid_north_pole_longitude
u_earth, v_earth = rotate_wind(u, v, lat, lon, pollat, pollon)

i = 100
print(checkfile.u_earth[-1, i, i])
print(u_earth[i, i])
print(checkfile.v_earth[-1, i, i])
print(v_earth[i, i])

# u1 = u[0, :, :]
# v1 = v[0, :, :]
# u_earth1 = u1 * arg2 * norm + v1 * arg1 * norm
# v_earth1 = -u1 * arg1 * norm + v1 * arg2 * norm
# i = 500
# np.array(u_earth1[i, i] - u1[i, i] * arg2[i, i] * norm[i, i] - \
#     v1[i, i] * arg1[i, i] * norm[i, i])
# u_earth = u * arg2 * norm + v * arg1 * norm
# v_earth = -u * arg1 * norm + v * arg2 * norm

# np.array(u_earth[0, i, i] - u_earth1[i, i])

'''

# endregion

# /project/pr94/qgao/miniconda3/envs/deoai/bin/python -c "from IPython import start_ipython; start_ipython()" --no-autoindent /project/pr94/qgao/DEoAI/DEoAI_analysis/python_codes/03_wind/3.1_rotate_wind_1km_100m.py





