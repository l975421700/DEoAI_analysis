

# =============================================================================
# region import packages


# basic library
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
    extent_global,
    extent12km_out,
    ticklabel1km,
    ticklabelm,
    ticklabelc,
    ticklabel12km,
    ticklabel1km_lb,
    ticklabel_global,
    ticklabel12km_out,
    transform,
    coastline,
    borders,
)

from DEoAI_analysis.module.statistics_calculate import(
    get_statistics
)

from DEoAI_analysis.module.spatial_analysis import(
    rotate_wind,
    sig_coeffs,
)


# endregion
# =============================================================================


# =============================================================================
# region global large scale environment plot

# plevl_201008 = xr.open_dataset(
#     'scratch/obs/era5/pressure_level_variables_20100801_09.nc')
plevl_201008_global = xr.open_dataset(
    'scratch/obs/era5/pressure_level_variables_20100801_09_global.nc')

lon = plevl_201008_global.longitude.values
lat = plevl_201008_global.latitude.values
plevel = 3
i_hour = 93
plevl_201008_global.time[i_hour]

tem = plevl_201008_global.t[i_hour, plevel, ].values - 273.15
geopotential = plevl_201008_global.z[i_hour, plevel, ].values
wind_u = plevl_201008_global.u[i_hour, plevel, ].values
wind_v = plevl_201008_global.v[i_hour, plevel, ].values

height = mpcalc.geopotential_to_height(
    geopotential * units('meter ** 2 / second ** 2'))
# stats.describe(tem.flatten())


fig, ax = framework_plot1("global")

# contour of geopotential height
# stats.describe(height.flatten()/10)
if plevel == 1:
    height_interval1 = np.arange(460, 600, 5)
    height_interval2 = np.arange(460, 600, 10)
if plevel == 3:
    height_interval1 = np.arange(30, 120, 5)
    height_interval2 = np.arange(30, 120, 10)
plt_height = ax.contour(
    lon, lat, height/10,
    colors='b', levels=height_interval1, linewidths=0.1,
    rasterized=True,
    )
ax.clabel(
    plt_height, inline=1, colors='b', fmt='%d',
    levels=height_interval2, inline_spacing=10, fontsize=8,
    )

# contour of temperature
# stats.describe(tem.flatten())
if plevel == 1:
    tem_interval1 = np.arange(-50, 4, 3)
    tem_interval2 = np.arange(-50, 4, 9)
if plevel == 3:
    tem_interval1 = np.arange(-50, 40, 3)
    tem_interval2 = np.arange(-50, 40, 9)
plt_tem = ax.contour(
    lon, lat, tem,
    colors='r', levels=tem_interval1, linewidths=0.1, linestyles = 'solid',
    rasterized=True,
    )
ax.clabel(
    plt_tem, inline=1, colors='r', fmt='%d',
    levels=tem_interval2, inline_spacing=10, fontsize=8,)

# wind barbs
# stats.describe(wind_u[::barb_interval, ::barb_interval].flatten())
barb_interval = 20
ax.barbs(lon[::barb_interval],
         lat[::barb_interval],
         wind_u[::barb_interval, ::barb_interval],
         wind_v[::barb_interval, ::barb_interval],
         length=2.5, barbcolor='black', linewidth=0.1,
         sizes={'spacing': 0.2, 'height': 0.4, 'width': 0.3, 'emptybarb': 0},
         rasterized=True,
         )

# fig.savefig('figures/00_test/trial.png', dpi=600)
fig.savefig(
    'figures/07_sst/7.6.1 global synoptic settings in ERA5_2010080621.png', dpi=1200)

# endregion
# =============================================================================


# =============================================================================
# region regional synoptic environment plevel plot

plev_201008 = xr.open_dataset(
    'scratch/obs/era5/plev_zuv_20100803_09_60_60.nc')

time = plev_201008.time.values
lon0 = plev_201008.longitude.values
lat0 = plev_201008.latitude.values
lon_range = np.where((lon0 >= 300) | (lon0 <= 0))[0]
lat_range = np.where((lat0 >= 0) & (lat0 <= 60))[0]
lon = lon0[lon_range]
lat = lat0[lat_range]
plevs = plev_201008.level.values

i_hour = 93
# plev_201008.time[i_hour]
plev = 975

plev_height = round(mpcalc.pressure_to_height_std(
    plev * units.hectopascal).to(units.meter).magnitude, 0)
# p0sl * np.exp(-(g * m * plev_height / (r0 * t0sl)))
i_plev = np.where(plevs == plev)[0][0]
geopotential = plev_201008.z[
    i_hour, i_plev, lat_range, lon_range].values
height = mpcalc.geopotential_to_height(
    geopotential * units('meter ** 2 / second ** 2')).magnitude
# stats.describe(height.flatten())

wind_u = plev_201008.u[
    i_hour, i_plev, lat_range, lon_range].values
wind_v = plev_201008.v[
    i_hour, i_plev, lat_range, lon_range].values

height_interval = 2
if plev == 500:
    height_interval = 2

height_interval1 = np.arange(
    np.floor(np.min(height/10) / height_interval - 1) * height_interval,
    np.ceil(np.max(height/10) / height_interval + 1) * height_interval,
    height_interval)

fig, ax = framework_plot1(
    "self_defined",
    dpi=600,
    extent=[-60, 0, 0, 60],
    ticklabel=ticks_labels(-60, 0, 0, 60, 10, 10),
    figsize=np.array([8.8, 9.5]) / 2.54,
    figure_margin={
        'left': 0.12, 'right': 0.97, 'bottom': 0.15, 'top': 0.995
    }
    )

# contour of geopotential height
plt_height = ax.contour(
    lon, lat, height/10,
    colors='b', levels=height_interval1, linewidths=0.2,)
ax_clabel = ax.clabel(
    plt_height, inline=1, colors='b', fmt='%d',
    levels=height_interval1, inline_spacing=10, fontsize=8,)
h1, _ = plt_height.legend_elements()
ax_legend = ax.legend([h1[0]],
                      ['Geopotential decameters at ' + str(plev) + 'hPa'],
                      loc='lower center', frameon=False,
                      bbox_to_anchor=(0.35, -0.18), handlelength=1,
                      columnspacing=1)

iarrow = 5
plt_quiver = ax.quiver(
    lon[::iarrow],
    lat[::iarrow],
    wind_u[::iarrow, ::iarrow],
    wind_v[::iarrow, ::iarrow],
    color='gray', rasterized=True, units='height', scale=500,
    width=0.002, headwidth=3, headlength=5, alpha=1)

ax.text(-45, -12,
        str(time[i_hour])[0:10] + ' ' + str(time[i_hour])[11:13] +':00 UTC',)

ax.quiverkey(plt_quiver, X=0.8, Y=-0.115, U=10,
            #  coordinates='data',
             label='10 [m/s]', labelpos='E', labelsep=0.05,)

fig.savefig(
    'figures/07_sst/7.6.5 synoptic env_2010080621 at ' + str(plev) + 'hPa.png',
    dpi=1200)



'''
'''
# endregion
# =============================================================================


# =============================================================================
# region regional synoptic environment plevel animation


plev_201008 = xr.open_dataset('scratch/obs/era5/plev_zuv_20100803_09_60_60.nc')

time = plev_201008.time.values
lon0 = plev_201008.longitude.values
lat0 = plev_201008.latitude.values
lon_range = np.where((lon0 >= 300) | (lon0 <= 0))[0]
lat_range = np.where((lat0 >= 0) & (lat0 <= 60))[0]
lon = lon0[lon_range]
lat = lat0[lat_range]
plevs = plev_201008.level.values

plev = 975
i_plev = np.where(plevs == plev)[0][0]
height_interval = 2
iarrow = 5

istart = 20  # 20
ifinal = 152  # 152

fig, ax = framework_plot1(
    "self_defined",
    dpi=600,
    extent=[-60, 0, 0, 60],
    ticklabel=ticks_labels(-60, 0, 0, 60, 10, 10),
    figsize=np.array([8.8, 9.5]) / 2.54,
    figure_margin={
        'left': 0.12, 'right': 0.97, 'bottom': 0.15, 'top': 0.995
    })
ims = []
for i_hour in np.arange(istart, ifinal):
    geopotential = plev_201008.z[
        i_hour, i_plev, lat_range, lon_range].values
    height = mpcalc.geopotential_to_height(
        geopotential * units('meter ** 2 / second ** 2')).magnitude
    wind_u = plev_201008.u[
        i_hour, i_plev, lat_range, lon_range].values
    wind_v = plev_201008.v[
        i_hour, i_plev, lat_range, lon_range].values
    height_interval1 = np.arange(
        np.floor(np.min(height/10) / height_interval - 1) * height_interval,
        np.ceil(np.max(height/10) / height_interval + 1) * height_interval,
        height_interval)
    
    plt_height = ax.contour(
        lon, lat, height/10,
        colors='b', levels=height_interval1, linewidths=0.2,)
    plt_height.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_height.collections
    ax_clabel = ax.clabel(
        plt_height, inline=1, colors='b', fmt='%d',
        levels=height_interval1, inline_spacing=10, fontsize=8,)
    
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=3, headlength=5, alpha=1)
    timetext = ax.text(-45, -12,
            str(time[i_hour])[0:10] + ' ' + str(time[i_hour])[11:13] + \
                ':00 UTC',)
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext])
    print(str(i_hour) + '/' + str(ifinal - 1))

h1, _ = plt_height.legend_elements()
ax.legend([h1[0]],
          ['Geopotential decameters at ' + str(plev) + 'hPa'],
          loc='lower center', frameon=False,
          bbox_to_anchor=(0.35, -0.18), handlelength=1,
          columnspacing=1)
ax.quiverkey(plt_quiver, X=0.8, Y=-0.115, U=10,
             label='10 [m/s]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=150)
ani.save(
    'figures/07_sst/7.6.5 synoptic env_20100803_09 at ' + str(plev) + 'hPa.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)




'''
# https://github.com/matplotlib/matplotlib/issues/6139
## Bug fix for Quad Contour set not having attribute 'set_visible'
# import types
# def setvisible(self,vis):
#     for c in self.collections: c.set_visible(vis)
# def setanimated(self,ani):
#     for c in self.collections: c.set_animated(ani)
    # plt_height.set_visible = types.MethodType(setvisible, plt_height)
    # plt_height.set_animated = types.MethodType(setanimated, plt_height)
    # plt_height.axes = plt.gca()
    # plt_height.figure = fig

'''
# endregion
# =============================================================================


# =============================================================================
# region regional synoptic environment single_level plot 201008

slev_201008 = xr.open_dataset(
    'scratch/obs/era5/siglelev_uvp_20100803_09_60_60.nc')
time = slev_201008.time.values
lon = slev_201008.longitude.values
lat = slev_201008.latitude.values

# i_hour = 20
# output_file = 'figures/07_sst/7.7.1 synoptic env_2010080320 at single level.png'
# i_hour = 93
# output_file = 'figures/07_sst/7.7.0 synoptic env_2010080621 at single level.png'
i_hour = 151
output_file = 'figures/07_sst/7.7.2 synoptic env_2010080907 at single level.png'
# slev_201008.time[i_hour]

pres = slev_201008.msl[i_hour, :, :].values / 100
# stats.describe(pres.flatten())
wind_u = slev_201008.u10[i_hour, :, :].values
wind_v = slev_201008.v10[i_hour, :, :].values

pres_interval = 3
pres_interval1 = np.arange(
    np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
    pres_interval)


def plot_maxmin_points(lon, lat, data, ax, extrema, nsize, symbol, color='k',
                       plotValue=False, transform=None):
    """
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    symbol: String 'H' or 'L'
    color: colors for 'H' or 'L' and values
    plot_value: Boolean, whether to plot the numeric value of max/min point
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
    import numpy as np
    
    data = gaussian_filter(data, sigma=3.0)
    # add dummy variables
    dummy = np.random.normal(0, 0.01, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    ny, nx = data.shape
    
    for i in range(len(mxy)):
        # 1st criterion
        criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
                     (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        if criteria1:
            ax.text(
                lon[mxx[i]], lat[mxy[i]], symbol,
                color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
        if (criteria1 & plotValue):
            ax.text(
                lon[mxx[i]], lat[mxy[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top')


fig, ax = framework_plot1(
    "self_defined",
    extent=[-60, 0, 0, 60],
    ticklabel=ticks_labels(-60, 0, 0, 60, 10, 10),
    figsize=np.array([8.8, 9.5]) / 2.54,
    figure_margin={
        'left': 0.12, 'right': 0.97, 'bottom': 0.15, 'top': 0.995})

# contour of geopotential height
plt_pres = ax.contour(
    lon, lat, pres,
    colors='b', levels=pres_interval1, linewidths=0.2,)
ax_clabel = ax.clabel(
    plt_pres, inline=1, colors='b', fmt='%d',
    levels=pres_interval1, inline_spacing=10, fontsize=8,)
h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend([h1[0]],
                      ['Mean sea level pressure [hPa]'],
                      loc='lower center', frameon=False,
                      bbox_to_anchor=(0.35, -0.18), handlelength=1,
                      columnspacing=1)

iarrow = 5
plt_quiver = ax.quiver(
    lon[::iarrow],
    lat[::iarrow],
    wind_u[::iarrow, ::iarrow],
    wind_v[::iarrow, ::iarrow],
    color='gray', rasterized=True, units='height', scale=500,
    width=0.002, headwidth=3, headlength=5, alpha=1)

ax.text(-45, -12,
        str(time[i_hour])[0:10] + ' ' + str(time[i_hour])[11:13] +':00 UTC',)

# Use definition to plot H/L symbols
plot_maxmin_points(lon, lat, pres, ax, 'max', 50, symbol='H', color='b')
plot_maxmin_points(lon, lat, pres, ax, 'min', 50, symbol='L', color='r')
ax.quiverkey(plt_quiver, X=0.8, Y=-0.115, U=10,
            #  coordinates='data',
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

fig.savefig(output_file, dpi=600)



'''
data = pres
extrema = 'max'
nsize = 9
symbol='H'
color='b'
plotValue=True
'''
# endregion
# =============================================================================


# =============================================================================
# region regional synoptic environment single_level animation 201008


def plot_maxmin_points(lon, lat, data, ax, extrema, nsize, symbol, color='k',
                       plotValue=False, transform=None):
    """
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    symbol: String 'H' or 'L'
    color: colors for 'H' or 'L' and values
    plot_value: Boolean, whether to plot the numeric value of max/min point
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
    import numpy as np
    
    data = gaussian_filter(data, sigma=3.0)
    # add dummy variables
    dummy = np.random.normal(0, 0.01, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    ny, nx = data.shape
    
    pretext = []
    for i in range(len(mxy)):
        # 1st criterion
        criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
                     (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        if criteria1:
            pretext_i = ax.text(
                lon[mxx[i]], lat[mxy[i]], symbol,
                color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
            pretext.append(pretext_i)
        if (criteria1 & plotValue):
            ax.text(
                lon[mxx[i]], lat[mxy[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top')
    
    return(pretext)


slev_201008 = xr.open_dataset(
    'scratch/obs/era5/siglelev_uvp_20100803_09_60_60.nc')
time = slev_201008.time.values
lon = slev_201008.longitude.values
lat = slev_201008.latitude.values

istart = 20  # 20
ifinal = 152  # 152

istart_var = np.where(time == np.datetime64('2010-08-06T05'))[0][0]
ifinal_var = np.where(time == np.datetime64('2010-08-07T19'))[0][0]
time[istart_var]
time[ifinal_var]


fig, ax = framework_plot1(
    "self_defined",
    dpi=600,
    extent=[-60, 0, 0, 60],
    ticklabel=ticks_labels(-60, 0, 0, 60, 10, 10),
    figsize=np.array([8.8, 9.5]) / 2.54,
    figure_margin={
        'left': 0.12, 'right': 0.97, 'bottom': 0.15, 'top': 0.995
    })

ims = []

for i_hour in np.arange(istart, ifinal):
    # i_hour = istart
    pres = slev_201008.msl[i_hour, :, :].values / 100
    wind_u = slev_201008.u10[i_hour, :, :].values
    wind_v = slev_201008.v10[i_hour, :, :].values
    
    pres_interval = 3
    pres_interval1 = np.arange(
        np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
        np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
        pres_interval)
    
    plt_pres = ax.contour(
        lon, lat, pres,
        colors='b', levels=pres_interval1, linewidths=0.2,)
    plt_pres.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_pres.collections
    
    ax_clabel = ax.clabel(
        plt_pres, inline=1, colors='b', fmt='%d',
        levels=pres_interval1, inline_spacing=10, fontsize=8,)
    
    iarrow = 5
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=3, headlength=5, alpha=1)
    
    hpre = plot_maxmin_points(
        lon, lat, pres, ax, 'max', 50, symbol='H', color='b')
    lpre = plot_maxmin_points(
        lon, lat, pres, ax, 'min', 50, symbol='L', color='r')
    
    timetext = ax.text(-45, -12,
            str(time[i_hour])[0:10] + ' ' + str(time[i_hour])[11:13] + \
                ':00 UTC',)
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext] + hpre + lpre)
    print(str(i_hour) + '/' + str(ifinal - 1))


h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend([h1[0]],
                      ['Mean sea level pressure [hPa]'],
                      loc='lower center', frameon=False,
                      bbox_to_anchor=(0.35, -0.18), handlelength=1,
                      columnspacing=1)
ax.quiverkey(plt_quiver, X=0.8, Y=-0.115, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=150)
ani.save(
    'figures/07_sst/7.7.3 synoptic env_20100803_09 at single level.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


# endregion
# =============================================================================


# =============================================================================
# region regional synoptic environment single_level plot general

slev_uvp = xr.open_dataset(
    'scratch/obs/era5/siglelev_uvp_201503_90_90.nc')
time = slev_uvp.time.values
lon = slev_uvp.longitude.values
lat = slev_uvp.latitude.values

i_hour = 1
output_file = 'figures/00_test/trial.png'
slev_uvp.time[i_hour]

pres = slev_uvp.msl[i_hour, :, :].values / 100
# stats.describe(pres.flatten())
wind_u = slev_uvp.u10[i_hour, :, :].values
wind_v = slev_uvp.v10[i_hour, :, :].values

pres_interval = 3
pres_interval1 = np.arange(
    np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
    pres_interval)


def plot_maxmin_points(lon, lat, data, ax, extrema, nsize, symbol, color='k',
                       plotValue=False, transform=None):
    """
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    symbol: String 'H' or 'L'
    color: colors for 'H' or 'L' and values
    plot_value: Boolean, whether to plot the numeric value of max/min point
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
    import numpy as np
    
    data = gaussian_filter(data, sigma=3.0)
    # add dummy variables
    dummy = np.random.normal(0, 0.01, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    ny, nx = data.shape
    
    for i in range(len(mxy)):
        # 1st criterion
        criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
                     (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        if criteria1:
            ax.text(
                lon[mxx[i]], lat[mxy[i]], symbol,
                color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
        if (criteria1 & plotValue):
            ax.text(
                lon[mxx[i]], lat[mxy[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top')


fig, ax = framework_plot1(
    "self_defined",
    extent=[-60, 0, 0, 60],
    ticklabel=ticks_labels(-60, 0, 0, 60, 10, 10),
    figsize=np.array([8.8, 9.5]) / 2.54,
    figure_margin={
        'left': 0.12, 'right': 0.97, 'bottom': 0.15, 'top': 0.995})

# contour of geopotential height
plt_pres = ax.contour(
    lon, lat, pres,
    colors='b', levels=pres_interval1, linewidths=0.2,)
ax_clabel = ax.clabel(
    plt_pres, inline=1, colors='b', fmt='%d',
    levels=pres_interval1, inline_spacing=10, fontsize=8,)
h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend([h1[0]],
                      ['Mean sea level pressure [hPa]'],
                      loc='lower center', frameon=False,
                      bbox_to_anchor=(0.35, -0.18), handlelength=1,
                      columnspacing=1)

iarrow = 5
plt_quiver = ax.quiver(
    lon[::iarrow],
    lat[::iarrow],
    wind_u[::iarrow, ::iarrow],
    wind_v[::iarrow, ::iarrow],
    color='gray', rasterized=True, units='height', scale=500,
    width=0.002, headwidth=3, headlength=5, alpha=1)

ax.text(-45, -12,
        str(time[i_hour])[0:10] + ' ' + str(time[i_hour])[11:13] +':00 UTC',)

# Use definition to plot H/L symbols
plot_maxmin_points(lon, lat, pres, ax, 'max', 50, symbol='H', color='b')
plot_maxmin_points(lon, lat, pres, ax, 'min', 50, symbol='L', color='r')
ax.quiverkey(plt_quiver, X=0.8, Y=-0.115, U=10,
            #  coordinates='data',
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

fig.savefig(output_file, dpi=600)


'''
data = pres
extrema = 'max'
nsize = 9
symbol='H'
color='b'
plotValue=True
'''
# endregion
# =============================================================================


# =============================================================================
# region regional synoptic environment single_level animation general

def plot_maxmin_points(lon, lat, data, ax, extrema, nsize, symbol, color='k',
                       plotValue=False, transform=None):
    """
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    symbol: String 'H' or 'L'
    color: colors for 'H' or 'L' and values
    plot_value: Boolean, whether to plot the numeric value of max/min point
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
    import numpy as np
    
    data = gaussian_filter(data, sigma=3.0)
    # add dummy variables
    dummy = np.random.normal(0, 0.01, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    ny, nx = data.shape
    
    pretext = []
    for i in range(len(mxy)):
        # 1st criterion
        criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
                     (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        if criteria1:
            pretext_i = ax.text(
                lon[mxx[i]], lat[mxy[i]], symbol,
                color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
            pretext.append(pretext_i)
        if (criteria1 & plotValue):
            ax.text(
                lon[mxx[i]], lat[mxy[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top')
    
    return(pretext)

slev_uvp = xr.open_dataset(
    'scratch/obs/era5/siglelev_uvp_201503_90_90.nc')
time = slev_uvp.time.values
lon = slev_uvp.longitude.values
lat = slev_uvp.latitude.values

istart = 0  # 20
ifinal = len(time)  # 2  #
output_file = 'figures/07_sst/7.7.4 synoptic env_201503 at single level.mp4'

fig, ax = framework_plot1(
    "self_defined",
    dpi=300,
    extent=[-60, 0, 0, 60],
    ticklabel=ticks_labels(-60, 0, 0, 60, 10, 10),
    figsize=np.array([8.8, 9.5]) / 2.54,
    figure_margin={
        'left': 0.12, 'right': 0.97, 'bottom': 0.15, 'top': 0.995
    })

ims = []

for i_hour in np.arange(istart, ifinal):
    # i_hour = istart
    
    pres = slev_uvp.msl[i_hour, :, :].values / 100
    wind_u = slev_uvp.u10[i_hour, :, :].values
    wind_v = slev_uvp.v10[i_hour, :, :].values
    pres_interval = 3
    pres_interval1 = np.arange(
        np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
        np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
        pres_interval)
    # stats.describe(pres.flatten())
    
    plt_pres = ax.contour(
        lon, lat, pres,
        colors='b', levels=pres_interval1, linewidths=0.2,)
    plt_pres.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_pres.collections
    
    ax_clabel = ax.clabel(
        plt_pres, inline=1, colors='b', fmt='%d',
        levels=pres_interval1, inline_spacing=10, fontsize=8,)
    
    iarrow = 5
    plt_quiver = ax.quiver(
        lon[::iarrow],
        lat[::iarrow],
        wind_u[::iarrow, ::iarrow],
        wind_v[::iarrow, ::iarrow],
        color='gray', rasterized=True, units='height', scale=500,
        width=0.002, headwidth=3, headlength=5, alpha=1)
    
    hpre = plot_maxmin_points(
        lon, lat, pres, ax, 'max', 50, symbol='H', color='b')
    lpre = plot_maxmin_points(
        lon, lat, pres, ax, 'min', 50, symbol='L', color='r')
    
    timetext = ax.text(-45, -12,
            str(time[i_hour])[0:10] + ' ' + str(time[i_hour])[11:13] + \
                ':00 UTC',)
    ims.append(add_arts + ax_clabel + [plt_quiver, timetext] + hpre + lpre)
    print(str(i_hour) + '/' + str(ifinal - 1))


h1, _ = plt_pres.legend_elements()
ax_legend = ax.legend([h1[0]],
                      ['Mean sea level pressure [hPa]'],
                      loc='lower center', frameon=False,
                      bbox_to_anchor=(0.35, -0.18), handlelength=1,
                      columnspacing=1)
ax.quiverkey(plt_quiver, X=0.8, Y=-0.115, U=10,
             label='10 [$m \; s^{-1}$]', labelpos='E', labelsep=0.05,)

ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=150)
ani.save(
    output_file,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

'''
# istart_var = np.where(time == np.datetime64('2010-08-06T05'))[0][0]
# ifinal_var = np.where(time == np.datetime64('2010-08-07T19'))[0][0]
# time[istart_var]
# time[ifinal_var]
'''
# endregion
# =============================================================================


# =============================================================================
# region regional synoptic environment single_level pdf plot general


def plot_maxmin_points(lon, lat, data, ax, extrema, nsize, symbol, color='k',
                       plotValue=False, transform=None):
    """
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    symbol: String 'H' or 'L'
    color: colors for 'H' or 'L' and values
    plot_value: Boolean, whether to plot the numeric value of max/min point
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
    import numpy as np
    
    data = gaussian_filter(data, sigma=3.0)
    # add dummy variables
    dummy = np.random.normal(0, 0.01, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    ny, nx = data.shape
    
    pretext = []
    for i in range(len(mxy)):
        # 1st criterion
        criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
                     (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        if criteria1:
            pretext_i = ax.text(
                lon[mxx[i]], lat[mxy[i]], symbol,
                color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
                horizontalalignment='center', verticalalignment='center')
            pretext.append(pretext_i)
        if (criteria1 & plotValue):
            ax.text(
                lon[mxx[i]], lat[mxy[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, clip_on=True, clip_box=ax.bbox,
                fontweight='bold', horizontalalignment='center',
                verticalalignment='top')
    
    return(pretext)

inputfile = 'scratch/obs/era5/siglelev_uvp_201503_90_90.nc'
outputfile = 'figures/07_sst/7.7.5 synoptic env_201503 at single level.pdf'

slev_uvp = xr.open_dataset(inputfile)
time = slev_uvp.time.values
lon = slev_uvp.longitude.values
lat = slev_uvp.latitude.values

istart = np.where(time == np.datetime64('2015-03-01T00:00'))[0][0]  # 20
ifinal = np.where(time == np.datetime64('2015-03-31T23:00'))[0][0] + 1  # 3  #

with PdfPages(outputfile) as pdf:
    for i_hour in np.arange(istart, ifinal):  # np.arange(istart, istart+5):  #
        ######## extract data
        pres = slev_uvp.msl[i_hour, :, :].values / 100
        wind_u = slev_uvp.u10[i_hour, :, :].values
        wind_v = slev_uvp.v10[i_hour, :, :].values
        pres_interval = 3
        pres_interval1 = np.arange(
            np.floor(np.min(pres) / pres_interval - 1) * pres_interval,
            np.ceil(np.max(pres) / pres_interval + 1) * pres_interval,
            pres_interval)
        
        fig, ax = framework_plot1(
            "self_defined",
            dpi=600,
            extent=[-60, 0, 0, 60],
            ticklabel=ticks_labels(-60, 0, 0, 60, 10, 10),
            figsize=np.array([8.8, 9.5]) / 2.54,
            figure_margin={
                'left': 0.12, 'right': 0.97, 'bottom': 0.15, 'top': 0.995},)
        
        plt_pres = ax.contour(
            lon, lat, pres,
            colors='b', levels=pres_interval1, linewidths=0.2,)
        plt_pres.__class__ = mpl.contour.QuadContourSet
        add_arts = plt_pres.collections
        ax_clabel = ax.clabel(
            plt_pres, inline=1, colors='b', fmt='%d',
            levels=pres_interval1, inline_spacing=10, fontsize=8,)
        
        iarrow = 5
        plt_quiver = ax.quiver(
            lon[::iarrow],
            lat[::iarrow],
            wind_u[::iarrow, ::iarrow],
            wind_v[::iarrow, ::iarrow],
            color='gray', rasterized=True, units='height', scale=500,
            width=0.002, headwidth=3, headlength=5, alpha=1)
        
        hpre = plot_maxmin_points(
            lon, lat, pres, ax, 'max', 50, symbol='H', color='b')
        lpre = plot_maxmin_points(
            lon, lat, pres, ax, 'min', 50, symbol='L', color='r')
        
        timetext = ax.text(
            -45, -12,
            str(time[i_hour])[0:10] + ' ' + str(time[i_hour])[11:13] + \
            ':00 UTC',)
        h1, _ = plt_pres.legend_elements()
        ax_legend = ax.legend(
            [h1[0]], ['Mean sea level pressure [hPa]'], loc='lower center',
            frameon=False, bbox_to_anchor=(0.35, -0.18), handlelength=1,
            columnspacing=1,)
        ax.quiverkey(
            plt_quiver, X=0.8, Y=-0.115, U=10, label='10 [$m \; s^{-1}$]',
            labelpos='E', labelsep=0.05, zorder = 2)
        # ax.set_rasterization_zorder(-10)
        pdf.savefig(fig)
        plt.close('all')
        print(str(i_hour) + '/' + str(ifinal - 1))

# endregion
# =============================================================================



