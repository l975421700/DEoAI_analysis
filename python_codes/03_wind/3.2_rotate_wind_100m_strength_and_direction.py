

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

from DEoAI_analysis.module.namelist import (
    month,
    seasons,
    months,
    years,
    years_months,
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
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
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
# region calculate strength and direction


folder_wind_1h_100m = 'scratch/wind_earth/wind_earth_1h_100m'


# np.arange(0, 10)
for i in np.arange(7, 10):
    begin_time = datetime.datetime.now()
    
    filelist_wind_1h_100m = \
        sorted(glob.glob(folder_wind_1h_100m + '/*100m20' + years[i] + '*'))
    
    wind_earth_1h_100m = xr.open_mfdataset(
        filelist_wind_1h_100m, concat_dim="time",
        data_vars='minimal', coords='minimal', compat='override'
    )
    
    time = wind_earth_1h_100m.time.data
    rlon = wind_earth_1h_100m.rlon.data
    rlat = wind_earth_1h_100m.rlat.data
    lon = wind_earth_1h_100m.lon.data
    lat = wind_earth_1h_100m.lat.data
    
    wind_earth_1h_100m_strength_direction = xr.Dataset(
        {"strength": (
            ("time", "rlat", "rlon"),
            np.zeros((len(time),
                      len(rlat) - 160,
                      len(rlon) - 160,
                      ))),
         "direction": (
             ("time", "rlat", "rlon"),
             np.zeros((len(time),
                       len(rlat) - 160,
                       len(rlon) - 160,
                       ))),
         "lat": (("rlat", "rlon"), lat[80:920, 80:920]),
         "lon": (("rlat", "rlon"), lon[80:920, 80:920]),
         },
        coords={
            "time": time,
            "rlat": rlat[80:920],
            "rlon": rlon[80:920],
        }
    )
    
    for j in np.arange(0, 24):
        indices = np.arange(j * len(time)/24, (j+1) *
                            len(time)/24, dtype='int64')
        u_earth = wind_earth_1h_100m.u_earth[indices, 80:920, 80:920].values
        print('u loaded    ' + str(j) + "/24" + str(datetime.datetime.now()))
        v_earth = wind_earth_1h_100m.v_earth[indices, 80:920, 80:920].values
        print('v loaded    ' + str(j) + "/24" + str(datetime.datetime.now()))
        
        wind_earth_1h_100m_strength_direction.strength[indices, :, :] = \
            (u_earth**2 + v_earth**2)**0.5
        print('strength calculated ' + str(j) +
              "/24 " + str(datetime.datetime.now()))
        
        wind_earth_1h_100m_strength_direction.direction[indices, :, :] = \
            mpcalc.wind_direction(u=u_earth * units('m/s'),
                                  v=v_earth * units('m/s'),
                                  convention='to')
        print('direction calculated    ' + str(j) +
              "/24" + str(datetime.datetime.now()))
    
    wind_earth_1h_100m_strength_direction.to_netcdf(
        "scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_20" +
        years[i] + ".nc"
    )
    print(str(i) + "/" + str(len(years)) + "   " +
          str(datetime.datetime.now() - begin_time))



'''
# check
wind_earth_1h_100m = xr.open_dataset(
        '/project/pr94/qgao/DEoAI/scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m201501.nc'
    )
wind_earth_1h_100m_strength_direction = xr.open_dataset('/project/pr94/qgao/DEoAI/scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2015.nc')

i = 20
print(np.array(wind_earth_1h_100m_strength_direction.strength[10, i, i]**2))
print(np.array(wind_earth_1h_100m.u_earth[10, i+80, i+80]**2 + \
                wind_earth_1h_100m.v_earth[10, i+80, i+80]**2))

print(wind_earth_1h_100m_strength_direction.direction[10, i, i].values)
print(wind_earth_1h_100m.u_earth[10, i+80, i+80].values)
print(wind_earth_1h_100m.v_earth[10, i+80, i+80].values)
np.tan(np.deg2rad(
    wind_earth_1h_100m_strength_direction.direction[10, i, i].values - 180)) - \
    (wind_earth_1h_100m.u_earth[10, i+80, i+80].values) / \
        (wind_earth_1h_100m.v_earth[10, i+80, i+80].values)

mpcalc.wind_direction(
    wind_earth_1h_100m.u_earth[10, i+80, i+80].values * units('m/s'),
    wind_earth_1h_100m.v_earth[10, i+80, i+80].values * units('m/s'),
    convention='to')

'''


# wind_earth_1h_100m_statistics = xr.Dataset(
#     {"wind_earth_quantiles": (
#         ("quantiles", "rlat", "rlon"),
#         np.zeros((len(quantiles[1]),
#                   len(rlat) - 160,
#                   len(rlon) - 160,
#                   ))),
#      "lat": (("rlat", "rlon"), lat[80:920, 80:920]),
#      "lon": (("rlat", "rlon"), lon[80:920, 80:920]),
#      },
#     coords={
#         "quantiles": quantiles[1],
#         "rlat": rlat[80:920],
#         "rlon": rlon[80:920],
#     }
# )

# q = quantiles[0]
# axis = 0

# # range(21)
# for i in range(21):
#     arr = np.array(rvorticity_1km_1h_100m.relative_vorticity[
#         :, np.arange(80, 120) + i * 40, 80:920])
#     rvorticity_1km_1h_100m_statistics.rvorticity_quantiles[
#         0, :, np.arange(0, 40) + i * 40, :] = get_statistics(
#         arr=arr, q=q, axis=axis
#     )


# endregion



# /project/pr94/qgao/miniconda3/envs/deoai/bin/python -c "from IPython import start_ipython; start_ipython()" --no-autoindent /project/pr94/qgao/DEoAI/DEoAI_analysis/python_codes/03_wind/3.1_rotate_wind_1km_100m.py





