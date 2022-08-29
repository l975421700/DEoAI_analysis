

# =============================================================================
# region import packages


# basic library
import datetime
import numpy as np
import xarray as xr
import os
import glob

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
    months,
    seasons,
    quantiles,
    timing,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

# endregion


# region calculation
folder_rvorticity_1km_1h_100m = \
    'scratch/relative_vorticity_1km_1h_100m'

filelist_rvorticity_1km_1h_100m = \
    sorted(glob.glob(folder_rvorticity_1km_1h_100m + '/*100m2010*'))


rvorticity_1km_1h_100m = xr.open_mfdataset(
    filelist_rvorticity_1km_1h_100m, concat_dim="time",
    data_vars='minimal', coords='minimal', parallel=True
)

time = rvorticity_1km_1h_100m.time

rvorticity_1km_1h_100m_average = xr.Dataset(
    {
        "m": (("time"), np.zeros(len(time))),
        "m_abs": (("time"), np.zeros(len(time))),
        "m_pos": (("time"), np.zeros(len(time))),
        "m_neg": (("time"), np.zeros(len(time))),
        "std": (("time"), np.zeros(len(time))),
        "std_abs": (("time"), np.zeros(len(time))),
        "std_pos": (("time"), np.zeros(len(time))),
        "std_neg": (("time"), np.zeros(len(time))),
    },
    coords={
        "time": time,
    }
)


# indices = np.arange(0, 8760)
indices = np.arange(0, 8760)

begin_time = datetime.datetime.now()

rvorticity_data = np.array(rvorticity_1km_1h_100m.relative_vorticity[
    indices, 80:920, 80:920])
del rvorticity_1km_1h_100m

print(str(datetime.datetime.now() - begin_time))

begin_time = datetime.datetime.now()
rvorticity_1km_1h_100m_average['m'][indices] = np.mean(
    rvorticity_data, axis=(1, 2)
)
rvorticity_1km_1h_100m_average['m_abs'][indices] = np.mean(
    abs(rvorticity_data), axis=(1, 2)
)

rvorticity_1km_1h_100m_average['m_pos'][indices] = np.nanmean(
    np.where(rvorticity_data > 0, rvorticity_data, np.nan), axis=(1, 2)
)
rvorticity_1km_1h_100m_average['m_neg'][indices] = np.nanmean(
    np.where(rvorticity_data < 0, rvorticity_data, np.nan), axis=(1, 2)
)

rvorticity_1km_1h_100m_average['std'][indices] = np.std(
    rvorticity_data, axis=(1, 2)
)
rvorticity_1km_1h_100m_average['std_abs'][indices] = np.std(
    abs(rvorticity_data), axis=(1, 2)
)

rvorticity_1km_1h_100m_average['std_pos'][indices] = np.nanstd(
    np.where(rvorticity_data > 0, rvorticity_data, np.nan), axis=(1, 2)
)
rvorticity_1km_1h_100m_average['std_neg'][indices] = np.nanstd(
    np.where(rvorticity_data < 0, rvorticity_data, np.nan), axis=(1, 2)
)

print(str(datetime.datetime.now() - begin_time))


rvorticity_1km_1h_100m_average.to_netcdf(
    "scratch/rvorticity_1km_1h_100m_average.nc"
)

# endregion


# region check calculation
# rvorticity_1km_1h_100m_average = xr.open_dataset(
#     "scratch/rvorticity_1km_1h_100m_average.nc"
# )
# i = 1005
# ddd = rvorticity_data

# print(rvorticity_1km_1h_100m_average['m'][i].values)
# print(np.mean(ddd[i, :, :]))

# print(rvorticity_1km_1h_100m_average['m_abs'][i].values)
# print(np.mean(np.absolute(ddd[i, :, :])))

# print(rvorticity_1km_1h_100m_average['m_pos'][i].values)
# print(np.mean(ddd[i, :, :][ddd[i, :, :] > 0]))

# print(rvorticity_1km_1h_100m_average['m_neg'][i].values)
# print(np.mean(ddd[i, :, :][ddd[i, :, :] < 0]))

# print(rvorticity_1km_1h_100m_average['std'][i].values)
# print(np.std(ddd[i, :, :]))

# print(rvorticity_1km_1h_100m_average['std_abs'][i].values)
# print(np.std(np.absolute(ddd[i, :, :])))

# print(rvorticity_1km_1h_100m_average['std_pos'][i].values)
# print(np.std(ddd[i, :, :][ddd[i, :, :] > 0]))

# print(rvorticity_1km_1h_100m_average['std_neg'][i].values)
# print(np.std(ddd[i, :, :][ddd[i, :, :] < 0]))

# endregion


# region visulization mean

rvorticity_1km_1h_100m_average = xr.open_dataset(
    "scratch/rvorticity/rvorticity_1km_1h_100m_average.nc"
)

# time = pd.date_range('2009-12-31', '2010-11-30', freq="1M")
time = [pd.to_datetime('2010-' + i + '-01',
                       infer_datetime_format=True) for i in months]

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8]) / 2.54)
# ax.set_extent([rvorticity_1km_1h_100m_average.time[0].values,
#                rvorticity_1km_1h_100m_average.time[-1].values,
#                0.5, 5.5])
plt_m, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average.m * 10**4,
    linewidth = 0.25, color = 'blue'
)

plt_m_abs, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average.m_abs * 10**4,
    linewidth = 0.25, color = 'lightskyblue'
)

plt_m_pos, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average.m_pos * 10**4,
    linewidth = 0.25, color = 'black'
)

plt_m_neg, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average.m_neg * 10**4,
    linewidth = 0.25, color = 'red'
)

ax_legend = ax.legend([plt_m, plt_m_abs, plt_m_pos, plt_m_neg],
           ['Mean over domain ', 'Mean of absolute values',
            'Mean of positive values', 'Mean of negative values'],
           loc = 'lower center', frameon = False, ncol = 2,
           bbox_to_anchor = (0.45, -0.375), handlelength = 1,
           columnspacing = 1)

for i in range(4):
    ax_legend.get_lines()[i].set_linewidth(1)

ax.set_xticks(time)
ax.set_yticks(np.arange(-5, 6))
ax.set_xticklabels(month, size=8)
ax.set_yticklabels(np.arange(-5, 6), size=8)
ax.set_xlabel('Year 2010', size=10)
ax.set_ylabel("Relative vorticity [$10^{-4}\;s^{-1}$]", size=10)
ax.grid(True, linewidth = 0.5, color = 'gray', alpha = 0.5, linestyle='--')

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.25, top=0.99)
fig.savefig(
    'figures/02_vorticity/2.2.0 rvorticity_in_2010.png', dpi=1200)

plt.close('all')
# endregion


# region visulization standard deviation

rvorticity_1km_1h_100m_average = xr.open_dataset(
    "scratch/rvorticity/rvorticity_1km_1h_100m_average.nc"
)

# time = pd.date_range('2009-12-31', '2010-11-30', freq="1M")
time = [pd.to_datetime('2010-' + i + '-01',
                       infer_datetime_format=True) for i in months]


fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8]) / 2.54)
# ax.set_extent([rvorticity_1km_1h_100m_average.time[0].values,
#                rvorticity_1km_1h_100m_average.time[-1].values,
#                0.5, 5.5])
plt_m, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average['std'] * 10**4,
    linewidth = 0.25, color = 'blue'
)

plt_m_abs, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average.std_abs * 10**4,
    linewidth = 0.25, color = 'lightskyblue'
)

plt_m_pos, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average.std_pos * 10**4,
    linewidth = 0.25, color = 'black'
)

plt_m_neg, = ax.plot(
    rvorticity_1km_1h_100m_average.time,
    rvorticity_1km_1h_100m_average.std_neg * 10**4,
    linewidth = 0.25, color = 'red'
)

ax_legend = ax.legend([plt_m, plt_m_abs, plt_m_pos, plt_m_neg],
           ['std over domain ', 'std of absolute values',
            'std of positive values', 'std of negative values'],
           loc = 'lower center', frameon = False, ncol = 2,
           bbox_to_anchor = (0.45, -0.375), handlelength = 1,
           columnspacing = 1)

for i in range(4):
    ax_legend.get_lines()[i].set_linewidth(1)

ax.set_xticks(time)
ax.set_yticks(np.arange(1, 9))
ax.set_xticklabels(month, size=8)
ax.set_yticklabels(np.arange(1, 9), size=8)
ax.set_xlabel('Year 2010', size=10)
ax.set_ylabel("Relative vorticity [$10^{-4}\;s^{-1}$]", size=10)
ax.grid(True, linewidth = 0.5, color = 'gray', alpha = 0.5, linestyle='--')

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.25, top=0.99)
fig.savefig(
    'figures/02_vorticity/2.2.1 rvorticity_std_in_2010.png', dpi=1200)

plt.close('all')
# endregion








