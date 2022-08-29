

# =============================================================================
# region import packages


# basic library
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.animation as animation

fontprop_tnr = fm.FontProperties(
    fname='/project/pr94/qgao/DEoAI/data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
mpl.rcParams['backend'] = 'Qt4Agg'  #
# mpl.get_backend()

plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

import cartopy as ctp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
)


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
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    months,
    years,
    years_months,
    month_days,
    timing,
    quantiles,
    folder_1km,
    g,
    m,
    r0,
    cp,
    r,
    r_v,
    p0sl,
    t0sl,
    extent1km,
    extent3d_m,
    extent3d_g,
    extent3d_t,
    extentm,
    extentc,
    extent12km,
    extent1km_lb,
    extent_global,
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
    ticklabel_global,
    transform,
    coastline,
    borders,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind
)


# endregion
# =============================================================================


# =============================================================================
# region daily pre in model simulation 1.1 km


filelist = np.array(sorted(glob.glob(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/6min_precip/day_lffd20*.nc')
))
# filelist[61:]

# ncfiles = xr.open_mfdataset(
#     filelist[61:], concat_dim="time",
#     data_vars='minimal', coords='minimal', compat='override'
# )

with xr.open_dataset(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/6min_precip/day_lffd20100220.nc') as ds:
    rlon = ds.rlon.data
    rlat = ds.rlat.data
    lon = ds.lon.data
    lat = ds.lat.data

# [80:920]
# [80:920, 80:920]

time = pd.date_range("2006-01-01", "2015-12-31", freq="1d")

daily_pre_1km_sim = xr.Dataset(
    {"daily_precipitation": (
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

for i in range(len(filelist[61:])):
    begin_time = datetime.datetime.now()
    # i = 0
    
    ncfile = xr.open_dataset(filelist[i + 61])
    
    if (time[i] != ncfile.time[0].values):
        print('Error: time indices do not match')
    
    daily_pre_1km_sim.daily_precipitation[i, :, :] = \
        ncfile.TOT_PREC[:, :, :].sum(axis=0).values
    
    if (i%10 == 0):
        print(str(i) + "/" + str(len(filelist[61:])) + "   " +
              str(datetime.datetime.now() - begin_time))

daily_pre_1km_sim.to_netcdf("scratch/precipitation/daily_pre_1km_sim.nc")


'''
dir1km = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/'
# nc1h = xr.open_dataset(dir1km + '1h/lffd20100220100000.nc')
nc1h_TOT_PREC = xr.open_dataset(dir1km + '1h_TOT_PREC/lffd20100220100000.nc')
nc6min_precip = xr.open_dataset(dir1km + '6min_precip/day_lffd20100220.nc')

# np.max(nc1h.TOT_PR.values) * 3600
np.max(nc1h_TOT_PREC.TOT_PREC.values)
np.max(nc6min_precip.TOT_PREC[91:101, :, :].sum(axis = 0))
# (nc1h_TOT_PREC.TOT_PREC.values[0, :, :] == nc6min_precip.TOT_PREC[91:101, :, :].sum(axis = 0)).all()

# check
daily_pre_1km_sim = xr.open_dataset("scratch/precipitation/daily_pre_1km_sim.nc")

ncfile = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/6min_precip/day_lffd20100219.nc')
np.max(np.abs(
    ncfile.TOT_PREC[:, :, :].sum(axis = 0).values - \
        daily_pre_1km_sim.daily_precipitation[
            np.where(daily_pre_1km_sim.time == np.datetime64('2010-02-19'))[0],
            :, :].values))
(ncfile.TOT_PREC[:, :, :].sum(axis = 0).values ==
    daily_pre_1km_sim.daily_precipitation[
        np.where(daily_pre_1km_sim.time == np.datetime64('2010-02-19'))[0],
        :, :].values).all()

# nc1h_TOT_PREC = xr.open_mfdataset(
#     '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h_TOT_PREC/lffd2010022[0-1]??0000.nc',
#     concat_dim="time", data_vars='minimal', coords='minimal', compat='override')
# np.max(np.abs(
#     nc1h_TOT_PREC.TOT_PREC[1:25, :, :].sum(axis = 0).values - \
#         daily_pre_1km_sim.daily_precipitation[
#             np.where(daily_pre_1km_sim.time == np.datetime64('2010-02-20'))[0],
#             :, :].values))

'''



# endregion
# =============================================================================


# =============================================================================
# region check


dir1km = '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/'
nc1h_TOT_PREC = xr.open_dataset(dir1km + '1h_TOT_PREC/lffd20100220110000.nc')
nc6min_precip = xr.open_dataset(dir1km + '6min_precip/day_lffd20100220.nc')

nc1h_TOT_PREC_mm = xr.open_dataset(
    dir1km + '1h_TOT_PREC_mm/lffd200601.nc')


# np.max(nc1h.TOT_PR.values) * 3600
np.max(nc1h_TOT_PREC.TOT_PREC.values)
np.max(nc6min_precip.TOT_PREC[101:111, :, :].sum(axis=0).values)


daily_pre_1km_sim = xr.open_dataset( "scratch/precipitation/daily_pre_1km_sim.nc")

ncfile = xr.open_dataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/6min_precip/day_lffd20100220.nc')
np.max(np.abs(
    ncfile.TOT_PREC[:, :, :].sum(axis=0).values -
    daily_pre_1km_sim.daily_precipitation[
        np.where(daily_pre_1km_sim.time == np.datetime64('2010-02-20'))[0],
        :, :].values))
(ncfile.TOT_PREC[:, :, :].sum(axis=0).values ==
    daily_pre_1km_sim.daily_precipitation[
        np.where(daily_pre_1km_sim.time == np.datetime64('2010-02-20'))[0],
        :, :].values).all()


# endregion
# =============================================================================


# =============================================================================
# region daily pre in model simulation 12 km

filelist = np.array(sorted(glob.glob(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h/lffd20*.nc')))
# np.where(filelist == '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h/lffd20060101000000.nc')
# np.where(filelist == '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h/lffd20151231230000.nc')
# filelist[52608:140256]

with xr.open_dataset(
        '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h/lffd20160101000000.nc') as ds:
    rlon = ds.rlon.data
    rlat = ds.rlat.data
    lon = ds.lon.data
    lat = ds.lat.data

time = pd.date_range("2006-01-01", "2015-12-31", freq="1d")

daily_pre_12km_sim = xr.Dataset(
    {"daily_precipitation": (
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

for i in range(int(len(filelist[52608:140256])/24)):
    begin_time = datetime.datetime.now()
    
    # i = 0
    ncfile = xr.open_mfdataset(
        filelist[52609 + 24 * i + np.arange(0, 24)], concat_dim="time",
        data_vars='minimal', coords='minimal', compat='override'
    )
    
    daily_pre_12km_sim.daily_precipitation[i, :, :] = \
        ncfile.TOT_PREC.sum(axis=0).values
    
    if (i % 10 == 0):
        print(str(i) + "/" + str(int(len(filelist[52608:140256])/24)) + "   " +
              str(datetime.datetime.now() - begin_time))

daily_pre_12km_sim.to_netcdf("scratch/precipitation/daily_pre_12km_sim.nc")



'''
# check

daily_pre_12km_sim = xr.open_dataset( "scratch/precipitation/daily_pre_12km_sim.nc")
filelist = np.array(sorted(glob.glob(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC12/lm_c/1h/lffd20*.nc')))

i = 100
filelist[52609 + 24 * i + np.arange(0, 24)]
ncfile = xr.open_mfdataset(
        filelist[52609 + 24 * i + np.arange(0, 24)], concat_dim="time",
        data_vars='minimal', coords='minimal', compat='override'
    )
(daily_pre_12km_sim.daily_precipitation[i, :, :].values == ncfile.TOT_PREC.sum(axis=0).values).all()

'''
# endregion
# =============================================================================

