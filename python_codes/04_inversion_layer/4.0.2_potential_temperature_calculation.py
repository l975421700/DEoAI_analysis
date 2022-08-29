

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
mpl.rcParams['backend'] = "Qt4Agg"
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

# 3D_Tenerife; 3D_GC
# np.arange(0, len(years))
for k in np.arange(0, 4):
    begin_time = datetime.datetime.now()
    print(begin_time)
    
    filelist_madeira_3d = sorted(glob.glob(
        folder_1km + '3D_Madeira/lfsd20' + years[k] + '*[0-9].nc'))
    madeira_3d = xr.open_mfdataset(
        filelist_madeira_3d, concat_dim="time", data_vars='minimal',
        coords='minimal', compat='override')
    print("data loaded  " + str(datetime.datetime.now() - begin_time))
    
    madeira_3d_p = xr.open_dataset(
        "scratch/3d/madeira/pressure/madeira_3d_p20" + years[k] + ".nc"
    )
    rlon = madeira_3d_p.rlon.data
    rlat = madeira_3d_p.rlat.data
    lon = madeira_3d_p.lon.data
    lat = madeira_3d_p.lat.data
    
    # create a file to store the results
    time = pd.date_range(
        "20" + years[k] + "-01-01-00",
        "20" + years[k] + "-12-31-23",
        freq="60min")
    madeira_3d_theta = xr.Dataset(
        {"theta": (
            ("time", "level", "rlat", "rlon"),
            np.zeros((len(time), len(madeira_3d.level.values),
                      len(rlat), len(rlon)))
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
    
    madeira_3d_theta.theta[:, :, :, :] = \
        madeira_3d.T.values * (p0sl/madeira_3d_p.P.values)**(r/cp)
    
    madeira_3d_theta.to_netcdf(
        "scratch/3d/madeira/theta/madeira_3d_theta" +
            "20" + years[k] + ".nc"
    )
    
    del madeira_3d_p, madeira_3d, madeira_3d_theta
    gc.collect()
    
    print(str(k) + "/" + str(len(years)) + "   " +
          str(datetime.datetime.now() - begin_time))


'''
# test
ddd = madeira_3d.T.values * (p0sl/madeira_3d_p.P[0:5, :, :, :].values)**(r/cp)

i = 2
j = 10
mpcalc.potential_temperature(
    madeira_3d_p.P[i, j, j, j].values / 100 * units.hPa,
    madeira_3d.T[i, j, j, j].values * units.kelvin)
madeira_3d.T[i, j, j, j].values * (p0sl/madeira_3d_p.P[i, j, j, j].values)**(r/cp)
ddd[i, j, j, j]


# check
# ddd = xr.open_dataset('scratch/3d/madeira/pressure/madeira_3d_p2015.nc')
# eee = xr.open_mfdataset(
#     '/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/3D_Madeira/lfsd20151231*0000.nc')

# i = 30
# j = -24
# print([ddd.P[j, i, i, i].values, eee.PP[j, i, i, i].values + p0[i, i, i]])

'''

# endregion
# =============================================================================






