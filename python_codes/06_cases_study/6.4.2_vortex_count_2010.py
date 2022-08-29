

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


smaller_domain = False
iyear = 4
# years[iyear]
# =============================================================================
# region import data

################################ wind data
wind_earth_1h_100m = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_20' + years[iyear] + '.nc')
time = wind_earth_1h_100m.time.values

################################ identified vortices
if smaller_domain:
    inputfile = "scratch/rvorticity/rvor_identify/re_identify/decades_past_sd/identified_transformed_rvor_20" + \
        years[iyear] + ".h5"
else:
    inputfile = "scratch/rvorticity/rvor_identify/re_identify/decades_past/identified_transformed_rvor_20" + \
    years[iyear] + ".h5"

identified_rvor = tb.open_file(inputfile, mode="r")
experiment = identified_rvor.root.exp1
hourly_vortex_count = experiment.vortex_info.cols.vortex_count[:]
# stats.describe(hourly_vortex_count)

# endregion
# =============================================================================


# =============================================================================
# region plot hourly time series in 2010 and check how many is isolated vortices

################################ filter hourly isolated vortices
previous_count = np.concatenate((np.array([0]), hourly_vortex_count[:-1]))
# (previous_count[1:] == hourly_vortex_count[:-1]).all()
next_count = np.concatenate((hourly_vortex_count[1:], np.array([0]), ))
# (next_count[:-1] == hourly_vortex_count[1:]).all()
# A series to store the number of hours that has vortices but no vortex in
# previous and next hour
isolated_hour = np.vstack(((hourly_vortex_count > 0), (previous_count == 0),
                           (next_count == 0))).all(axis = 0)

filtered_hourly_vortex_count = hourly_vortex_count.copy()
filtered_hourly_vortex_count[isolated_hour] = 0

################################ calculate daily average
# calculate rolling mean
pd_hourly_vortex_count = pd.Series(
    data = filtered_hourly_vortex_count, index = time)
pd_hourly_vortex_count_rolling_mean = \
    pd_hourly_vortex_count.rolling(24, min_periods=1).mean()
pd_daily_vortex_count = pd_hourly_vortex_count.resample('1D').sum()
# pd_daily_vortex_count_rolling_mean = \
#     pd_daily_vortex_count.rolling(10, min_periods=1).mean()
pd_daily_vortex_count_rolling_mean = \
    pd_daily_vortex_count.rolling(10, min_periods=1, center = True).mean()

################################ plot daily time series

date = pd.date_range("2010-01-01", '2010-12-31', freq="M")

# date = [pd.to_datetime('20' + years[iyear] + '-' + i + '-01',
#                        infer_datetime_format=True) for i in months]

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54, dpi=600)

plt_mean1, = ax.plot(
    pd_daily_vortex_count.index,
    pd_daily_vortex_count.values, linewidth=0.5, color='black'
)

plt_mean2, = ax.plot(
    pd_daily_vortex_count_rolling_mean.index,
    pd_daily_vortex_count_rolling_mean.values, linewidth=1, color='red'
)

ax.set_xticks(date)
ax.set_xticklabels(month, size=8)
ax.set_yticks(np.arange(0, 110, 20))
ax.set_yticklabels(np.arange(0, 110, 20), size=8)
ax.set_xlabel('Year 20' + years[iyear], size=10)
ax.set_ylabel("Daily count of vortices [#]", size=10)
ax.grid(True, linewidth=0.25, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.14, right=0.99, bottom=0.14, top=0.99)

if smaller_domain:
    fig.savefig(
        'figures/09_decades_vortex/09_07_smaller_domain/9_7.1.0_daily_vortex_count_in_2010_sd.png')
else:
    fig.savefig(
        'figures/09_decades_vortex/9.0.2 daily vortex count in 2010.png')



################################ plot hourly time series
# date = [pd.to_datetime('20' + years[iyear] + '-' + i + '-01',
#                        infer_datetime_format=True) for i in months]

# fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8]) / 2.54)

# plt_mean1, = ax.plot(
#     time, hourly_vortex_count, linewidth=0.25, color='black'
# )
# plt_mean3, = ax.plot(
#     time, filtered_hourly_vortex_count, linewidth=0.25, color='blue'
# )
# plt_mean2, = ax.plot(
#     time, pd_hourly_vortex_count_rolling_mean, linewidth=0.25, color='red'
# )

# ax_legend = ax.legend([plt_mean1, plt_mean2],
#                       ['Hourly vortex count ', '24 hours running mean'],
#                       loc='lower center', frameon=False, ncol=2,
#                       bbox_to_anchor=(0.45, -0.275), handlelength=1,
#                       columnspacing=1)
# for i in range(2):
#     ax_legend.get_lines()[i].set_linewidth(1)

# ax.set_xticks(date)
# ax.set_xticklabels(month, size=8)
# ax.set_yticks(np.arange(0, 7, 1))
# ax.set_yticklabels(np.arange(0, 7, 1), size=8)
# ax.set_xlabel('Year 20' + years[iyear], size=10)
# ax.set_ylabel("Number of vortices [#]", size=10)
# ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# fig.subplots_adjust(left=0.12, right=0.99, bottom=0.2, top=0.98)
# fig.savefig(
#     'figures/09_decades_vortex/9.0.1 madeira vortex count in 2010.png', dpi=600)

# plt.close('all')

'''
time[np.where(hourly_vortex_count == np.max(hourly_vortex_count))]
'''
# endregion
# =============================================================================




