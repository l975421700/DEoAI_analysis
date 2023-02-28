

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

fontprop_tnr = fm.FontProperties(
    fname='/project/pr94/qgao/DEoAI/data_source/TimesNewRoman.ttf')
# mpl.rcParams['font.family'] = fontprop_tnr.get_name()
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=10)
# mpl.rcParams['backend'] = "Qt4Agg"
# mpl.get_backend()

plt.rcParams.update({"mathtext.fontset": "stix"})
plt.rcParams["font.serif"] = ["Times New Roman"]

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
    g,
    m,
    r0,
    cp,
    r,
    r_v,
    p0sl,
    t0sl,
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
# region madeira pressure calculation

with xr.open_dataset(
        folder_1km + '3D_Madeira/lfsd20051101000000c.nc') as madeira_3d_const:
    rlon = madeira_3d_const.rlon.data
    rlat = madeira_3d_const.rlat.data
    lon = madeira_3d_const.lon.data
    lat = madeira_3d_const.lat.data
    hhl = madeira_3d_const.HHL.squeeze()
    hhl_level = (hhl[:-1, :, :] + hhl[1:, :, :])/2
    p0 = p0sl * np.exp(-(g * m * hhl_level / (r0 * t0sl))).values

# np.arange(0, len(years))
for k in [9]:
    begin_time = datetime.datetime.now()
    print(begin_time)
    filelist_madeira_3d = sorted(glob.glob(
        folder_1km + '3D_Madeira/lfsd20' + years[k] + '*[0-9].nc'))
    madeira_3d = xr.open_mfdataset(
        filelist_madeira_3d, data_vars='minimal',
        coords='minimal', compat='override')
    print("data loaded  " + str(datetime.datetime.now() - begin_time))
    
    # create a file to store the results
    time = pd.date_range(
        "20" + years[k] + "-01-01-00",
        "20" + years[k] + "-12-31-23",
        freq="60min")
    madeira_3d_p = xr.Dataset(
        {"P": (
            ("time", "level", "rlat", "rlon"),
            np.zeros((len(time), len(madeira_3d.level.values), len(rlat), len(rlon)))
            ),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon),
         },
        coords={
            "time": time,
            "level": madeira_3d.level.values,
            "rlat": rlat,
            "rlon": rlon,
        }
    )
    
    madeira_3d_p.P[:, :, :, :] = madeira_3d.PP.values + p0
    
    madeira_3d_p.to_netcdf(
        "scratch/3d/madeira/pressure/madeira_3d_p" + "20" + years[k] + ".nc"
    )
    
    del madeira_3d_p, madeira_3d
    gc.collect()
    
    print(str(k) + "/" + str(len(years)) + "   " +
          str(datetime.datetime.now() - begin_time))


'''
# check
i = 20
j = 59
print([hhl_level[j, i, i].values, (hhl[j, i, i].values + hhl[j + 1, i, i].values)/2])

p0[59, :, :]

p500 = file_c_3d_madeira.vcoord.p0sl * \
    np.exp(-(g * m * 20000 / (r0 * file_c_3d_madeira.vcoord.t0sl)))
p500metpy = mpcalc.add_height_to_pressure(
    file_c_3d_madeira.vcoord.p0sl * units('Pa'),
    20000  * units('m'))

# test
ddd = madeira_3d.PP.values + p0
i = 25
print([ddd[4, i, i, i], madeira_3d.PP[4, i, i, i].values + p0[i, i, i]])

# check
ddd = xr.open_dataset('scratch/3d/madeira/pressure/madeira_3d_p2015.nc')
eee = xr.open_mfdataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20151231*0000.nc')

i = 30
j = -24
print([ddd.P[j, i, i, i].values, eee.PP[j, i, i, i].values + p0[i, i, i]])

'''

# endregion
# =============================================================================


# =============================================================================
# region tenerife pressure calculation

# 3D_GC

with xr.open_dataset(
        folder_1km + '3D_Tenerife/lfsd20051101000000c.nc') as tenerife_3d_const:
    rlon = tenerife_3d_const.rlon.data
    rlat = tenerife_3d_const.rlat.data
    lon = tenerife_3d_const.lon.data
    lat = tenerife_3d_const.lat.data
    hhl = tenerife_3d_const.HHL.squeeze()
    hhl_level = (hhl[:-1, :, :] + hhl[1:, :, :])/2
    p0 = p0sl * np.exp(-(g * m * hhl_level / (r0 * t0sl))).values

# np.arange(0, len(years))
for k in np.arange(8, len(years)):
    begin_time = datetime.datetime.now()
    print(begin_time)
    filelist_tenerife_3d = sorted(glob.glob(
        folder_1km + '3D_Tenerife/lfsd20' + years[k] + '*[0-9].nc'))
    tenerife_3d = xr.open_mfdataset(
        filelist_tenerife_3d, data_vars='minimal',
        coords='minimal', compat='override')
    print("data loaded  " + str(datetime.datetime.now() - begin_time))
    
    # create a file to store the results
    time = pd.date_range(
        "20" + years[k] + "-01-01-00",
        "20" + years[k] + "-12-31-23",
        freq="60min")
    tenerife_3d_p = xr.Dataset(
        {"P": (
            ("time", "level", "rlat", "rlon"),
            np.zeros(
                (len(time), len(tenerife_3d.level.values), len(rlat), len(rlon)))
            ),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon),
         },
        coords={
            "time": time,
            "level": tenerife_3d.level.values,
            "rlat": rlat,
            "rlon": rlon,
        }
    )
    
    tenerife_3d_p.P[:, :, :, :] = tenerife_3d.PP.values + p0
    
    tenerife_3d_p.to_netcdf(
        "scratch/3d/tenerife/pressure/tenerife_3d_p" + "20" + years[k] + ".nc"
    )
    
    del tenerife_3d_p, tenerife_3d
    gc.collect()
    
    print(str(k) + "/" + str(len(years)) + "   " +
          str(datetime.datetime.now() - begin_time))


'''
# check
ddd = xr.open_dataset('scratch/3d/tenerife/pressure/tenerife_3d_p2010.nc')
eee = xr.open_mfdataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Tenerife/lfsd20101231*0000.nc')

i = 20
j = -5
print([ddd.P[j, i, i, i].values, eee.PP[j, i, i, i].values + p0[i, i, i]])

'''

# endregion
# =============================================================================


# =============================================================================
# region gran canary pressure calculation

with xr.open_dataset(
        folder_1km + '3D_GC/lfsd20051101000000c.nc') as canary_3d_const:
    rlon = canary_3d_const.rlon.data
    rlat = canary_3d_const.rlat.data
    lon = canary_3d_const.lon.data
    lat = canary_3d_const.lat.data
    hhl = canary_3d_const.HHL.squeeze()
    hhl_level = (hhl[:-1, :, :] + hhl[1:, :, :])/2
    p0 = p0sl * np.exp(-(g * m * hhl_level / (r0 * t0sl))).values

# np.arange(0, len(years))
for k in np.arange(8, len(years)):
    begin_time = datetime.datetime.now()
    print(begin_time)
    filelist_canary_3d = sorted(glob.glob(
        folder_1km + '3D_GC/lfsd20' + years[k] + '*[0-9].nc'))
    canary_3d = xr.open_mfdataset(
        filelist_canary_3d, data_vars='minimal',
        coords='minimal', compat='override')
    print("data loaded  " + str(datetime.datetime.now() - begin_time))
    
    # create a file to store the results
    time = pd.date_range(
        "20" + years[k] + "-01-01-00",
        "20" + years[k] + "-12-31-23",
        freq="60min")
    canary_3d_p = xr.Dataset(
        {"P": (
            ("time", "level", "rlat", "rlon"),
            np.zeros(
                (len(time), len(canary_3d.level.values), len(rlat), len(rlon)))
            ),
         "lat": (("rlat", "rlon"), lat),
         "lon": (("rlat", "rlon"), lon),
         },
        coords={
            "time": time,
            "level": canary_3d.level.values,
            "rlat": rlat,
            "rlon": rlon,
        }
    )
    
    canary_3d_p.P[:, :, :, :] = canary_3d.PP.values + p0
    
    canary_3d_p.to_netcdf(
        "scratch/3d/canary/pressure/canary_3d_p" + "20" + years[k] + ".nc"
    )
    
    del canary_3d_p, canary_3d
    gc.collect()
    
    print(str(k) + "/" + str(len(years)) + "   " +
          str(datetime.datetime.now() - begin_time))


'''
# check
ddd = xr.open_dataset('scratch/3d/canary/pressure/canary_3d_p2010.nc')
eee = xr.open_mfdataset(
    '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_GC/lfsd20101231*0000.nc')

i = 30
j = -5
print([ddd.P[j, i, i, i].values, eee.PP[j, i, i, i].values + p0[i, i, i]])

'''

# endregion
# =============================================================================




