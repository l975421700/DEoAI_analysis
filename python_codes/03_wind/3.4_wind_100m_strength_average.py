

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
# region calculate

wind_earth_1h_100m_strength_direction = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

time = wind_earth_1h_100m_strength_direction.time

wind_earth_1h_100m_average = xr.Dataset(
    {
        "spatial_quantiles": (("quantiles", "time"),
                              np.zeros((len(quantiles[1]),
                                        len(time),
                                        ))),
    },
    coords={
        "quantiles": quantiles[1],
        "time": time,
    }
)

arr = wind_earth_1h_100m_strength_direction.strength[
    :, :, :].values
wind_earth_1h_100m_average.spatial_quantiles[:, :] = get_statistics(
    arr=arr, q=quantiles[0], axis=(1, 2)
)

wind_earth_1h_100m_average.to_netcdf(
    "scratch/wind_earth/wind_earth_1h_100m_average_2010.nc"
)

'''
# test
arr = wind_earth_1h_100m_strength_direction.strength[
    0:5, :, :].values
ddd = get_statistics(
    arr=arr, q=quantiles[0], axis=(1, 2)
)

# check
ddd = xr.open_dataset('scratch/wind_earth/wind_earth_1h_100m_average_2010.nc')
eee = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

i = -100
print(ddd.spatial_quantiles[0, i])
print(np.min(eee.strength[i, :, :].values))
print(ddd.spatial_quantiles[8, i])
print(np.max(eee.strength[i, :, :].values))

print(ddd.spatial_quantiles[13, i])
print(np.mean(eee.strength[i, :, :].values))
print(ddd.spatial_quantiles[14, i])
print(np.std(eee.strength[i, :, :].values))

'''

# endregion


# =============================================================================
# region plot


wind_earth_1h_100m_average = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m_average_2010.nc')


# time = pd.date_range('2009-12-31', '2010-11-30', freq="1M")
time = [pd.to_datetime('2010-' + i + '-01',
                       infer_datetime_format=True) for i in months]

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 8]) / 2.54)

plt_mean1, = ax.plot(
    wind_earth_1h_100m_average.time,
    wind_earth_1h_100m_average.spatial_quantiles[13, :],
    linewidth=0.25, color='black'
)

spatial_quantiles_mean = pd.Series(
    wind_earth_1h_100m_average.spatial_quantiles[13, :].values)

spatial_quantiles_rolling_mean = np.array(spatial_quantiles_mean.rolling(
    24*10, min_periods=1).mean())
plt_mean2, = ax.plot(
    wind_earth_1h_100m_average.time,
    spatial_quantiles_rolling_mean,
    linewidth=0.75, color='red'
)

'''
check
i=10
print(np.mean(spatial_quantiles_mean[0:i+1]))
print(spatial_quantiles_rolling_mean[i])

i=1000
print(np.mean(spatial_quantiles_mean[i-23:i+1]))
print(spatial_quantiles_rolling_mean[i])

'''

ax_legend = ax.legend([plt_mean1, plt_mean2],
                      ['Mean over domain ', '10 days running mean'],
                      loc='lower center', frameon=False, ncol=2,
                      bbox_to_anchor=(0.45, -0.275), handlelength=1,
                      columnspacing=1)

for i in range(2):
    ax_legend.get_lines()[i].set_linewidth(1)

ax.set_xticks(time)
ax.set_yticks(np.arange(2, 18, 2))
ax.set_xticklabels(month, size=8)
ax.set_yticklabels(np.arange(2, 18, 2), size=8)
ax.set_xlabel('Year 2010', size=10)
ax.set_ylabel("Wind velocity [m/s]", size=10)
ax.grid(True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.2, top=0.98)
fig.savefig(
    'figures/03_wind/3.1.0 velocity_in_2010.png', dpi=1200)

plt.close('all')

# endregion





