

# =============================================================================
# region import packages


# basic library
from matplotlib.path import Path
import pywt.data
import pywt
import datetime
import numpy as np
import xarray as xr
import os
import glob
import pickle
import gc
from scipy import stats
import tables as tb

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
# Background image
os.environ["CARTOPY_USER_BACKGROUNDS"] = "data_source/share/bg_cartopy"

from DEoAI_analysis.module.mapplot import (
    ticks_labels,
    scale_bar,
    framework_plot,
    framework_plot1,
)

from DEoAI_analysis.module.vortex_namelist import (
    correctly_reidentified,
    track_p1, track_p2, track_p3, track_p4,
    track_n1, track_n2, track_n3, track_n4,
)

# data analysis
import pandas as pd
import xesmf as xe
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units
from haversine import haversine
from scipy import interpolate
from scipy import ndimage
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
import h5py
from scipy.ndimage.filters import median_filter, maximum_filter
from haversine import haversine_vector, Unit

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
    rvor_level,
    rvor_ticks,
    rvor_cmp,
    center_madeira,
    angle_deg_madeira,
    radius_madeira,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    sig_coeffs,
    vortex_identification,
    vortex_identification1,
)


# endregion
# =============================================================================


# =============================================================================
# region create mask for ellipse 3

nc3d_lb_c = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')

lon = nc3d_lb_c.lon.values
lat = nc3d_lb_c.lat.values

from math import sin, cos
def ellipse(h, k, a, b, phi, x, y):
    xp = (x-h)*cos(phi) + (y-k)*sin(phi)
    yp = -(x-h)*sin(phi) + (y-k)*cos(phi)
    return (xp/a)**2 + (yp/b)**2 <= 1

mask_e3 = ellipse(
    center_madeira[0] + radius_madeira[1] * 5 * np.cos(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    center_madeira[1] - radius_madeira[1] * 5 * np.sin(
        np.pi/2 - np.deg2rad(angle_deg_madeira)),
    radius_madeira[0], radius_madeira[1],
    np.deg2rad(angle_deg_madeira), lon, lat
    )

# endregion
# =============================================================================


# =============================================================================
# region import data
iyear = 4
# years[iyear]

################################ 100m vorticity
# rvor_100m_f = np.array(sorted(glob.glob(
#     'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
# rvor_100m = xr.open_mfdataset(
#     rvor_100m_f, concat_dim="time", data_vars='minimal',
#     coords='minimal', compat='override', chunks={'time': 1})
# time = rvor_100m.time.values
# lon = rvor_100m.lon[80:920, 80:920].values
# lat = rvor_100m.lat[80:920, 80:920].values

################################ import wind
wind_100m_sd = xr.open_mfdataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_20' + years[iyear] + '.nc',
    concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = wind_100m_sd.time.values
lon = wind_100m_sd.lon.values
lat = wind_100m_sd.lat.values

# endregion
# =============================================================================


# =============================================================================
# region calculate ratio of advection velocity to wind velocity at e_3!!!

# initial = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
inputfile = 'scratch/rvorticity/rvor_track/identified_transformed_rvor_20100803_09_track_4p_4n.h5'

################################ import tracked vortices
tracked_rvor = tb.open_file(inputfile, mode="r")
group_name = ['P1', 'P2', 'P3', 'P4', 'N1', 'N2', 'N3', 'N4']
# 12+ 12+ 12+ 14+ 16+ 15+ 17+ 12 - 8

################################ create array to store velocity
distance_interval = np.array([])
time_interval = np.array([])
mean_wind = np.array([])


################################ extract velocity from each tracked vortices
for j in range(len(group_name)):
    # j = 0
    iexp = tracked_rvor.root[group_name[j]]
    center_lat = iexp.extracted_vortex_info.cols.center_lat[:]
    center_lon = iexp.extracted_vortex_info.cols.center_lon[:]
    time_index = iexp.extracted_vortex_info.cols.time_index[:]
    
    # time[time_index]
    
    jmean_wind1 = wind_100m_sd.strength[time_index, ].values[:, mask_e3].mean(axis=1)
    jmean_wind = (jmean_wind1[1:] + jmean_wind1[:-1])/2
    # mean_wind_u = iexp.extracted_vortex_info.cols.mean_wind_u[:]
    # mean_wind_v = iexp.extracted_vortex_info.cols.mean_wind_v[:]
    # jmean_wind1 = np.sqrt(mean_wind_u ** 2 + mean_wind_v ** 2)
    # jmean_wind = (jmean_wind1[1:] + jmean_wind1[:-1])/2
    
    jdistance_interval = haversine_vector(
        [tuple(i) for i in zip(center_lat[:-1], center_lon[:-1])],
        [tuple(i) for i in zip(center_lat[1:], center_lon[1:])],
        Unit.METERS)
    jtime_interval = (time[time_index[1:]] -
                     time[time_index[:-1]]) / np.timedelta64(1, 's')
    distance_interval = np.concatenate((distance_interval, jdistance_interval))
    time_interval = np.concatenate((time_interval, jtime_interval))
    mean_wind = np.concatenate((mean_wind, jmean_wind))

advection_velocity = distance_interval / time_interval
stats.describe(advection_velocity/mean_wind)
# DescribeResult(nobs=102, minmax=(0.8465939810490709, 1.535713537478737), mean=1.031747010500003, variance=0.008618248660375322, skewness=1.8364618743064465, kurtosis=7.611413102357822)

tracked_rvor.close()


'''
'''
# endregion
# =============================================================================






