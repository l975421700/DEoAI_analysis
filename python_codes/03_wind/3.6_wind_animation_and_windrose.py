

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
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
# region wind animation

wind_earth_1h_100m = xr.open_dataset('scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

lon = wind_earth_1h_100m.lon.values
lat = wind_earth_1h_100m.lat.values
time = wind_earth_1h_100m.time.values

# set colormap level and ticks
windlevel = np.arange(0, 15.1, 0.5)
ticks = np.arange(0, 15.1, 3)

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ims = []
istart = 5156  # 5156
ifinal = 5288  # 5158

for i in np.arange(istart, ifinal):
    
    wind = wind_earth_1h_100m.strength[i, :, :].values
    
    plt_wind = ax.pcolormesh(
        lon, lat, wind, cmap=cm.get_cmap('RdYlBu_r', 31),
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        transform=transform, rasterized=True,
        # zorder=-2,
        )
    
    wind_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC',
        )
    
    ims.append([plt_wind, wind_time])
    print(str(i) + '/' + str(ifinal - 1))

gl = ax.gridlines(crs=transform, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.add_feature(borders); ax.add_feature(coastline)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Wind velocity [m/s]")

scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )

ax.set_extent(extent1km_lb, crs=transform)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save('figures/03_wind/3.4.0_Wind_in_2010080320_2010080907.mp4')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region wind strength and direction visulization

wind_earth = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m201008.nc')

wind_earth_1h_100m = xr.open_dataset('scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

lon = wind_earth_1h_100m.lon.values
lat = wind_earth_1h_100m.lat.values
time = wind_earth_1h_100m.time.values

windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)

istart = 5287  # 5156
ifinal = 5288  # 5158

fig, ax = framework_plot("1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,)
# ax.set_facecolor(plt.cm.Greys(0.5))

wind = wind_earth_1h_100m.strength[istart, :, :].values
plt_wind = ax.pcolormesh(
    lon, lat, wind, cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    transform=transform, rasterized=True,
    # zorder=-2,
)
cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Wind velocity [m/s]")

iarrow = 30
ax.quiver(
    lon[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
    lat[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
    wind_earth.u_earth[istart-5088, (80 + int(iarrow/2)):920:iarrow,
                       (80 + int(iarrow/2)):920:iarrow] / \
                           (wind[int(iarrow/2)::iarrow,
                                 int(iarrow/2)::iarrow]),
    wind_earth.v_earth[istart-5088, (80 + int(iarrow/2)):920:iarrow,
                       (80 + int(iarrow/2)):920:iarrow] / \
                           (wind[int(iarrow/2)::iarrow,
                                 int(iarrow/2)::iarrow]),
    # wind[int(iarrow/2)::iarrow, int(iarrow/2)::iarrow],
    # cmap=cm.get_cmap('RdYlBu_r', len(windlevel)),
    # norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
    # scale=80,
    # headlength=20,
    # headwidth=12,
    # width=0.004,
    # alpha=0.5,
)
wind_time = ax.text(
    -23, 34, str(time[istart])[0:10] + ' ' + \
    str(time[istart])[11:13] + ':00 UTC',
)
scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
fig.savefig('figures/03_wind/3.7.1 wind strength and direction in 2010-08-09 07:00 UTC.png', dpi=600)
plt.close('all')

'''

'''

# endregion
#  =============================================================================


# =============================================================================
# region wind strength and direction animation

######## mask topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

######## mask analysis region outside
# create rectangle for analysis region
daily_pre_1km_sim = xr.open_dataset(
    "scratch/precipitation/daily_pre_1km_sim.nc")
lon1 = daily_pre_1km_sim.lon.values
lat1 = daily_pre_1km_sim.lat.values
analysis_region = np.zeros_like(lon1)
analysis_region[80:920, 80:920] = 1
cs = plt.contour(lon1, lat1, analysis_region, levels=np.array([0.5]))
poly_path = cs.collections[0].get_paths()[0]
mask_lon = np.arange(-23.5, -11.2, 0.02)
mask_lat = np.arange(24.1, 34.9, 0.02)
mask_lon2, mask_lat2 = np.meshgrid(mask_lon, mask_lat)
coors = np.hstack((mask_lon2.reshape(-1, 1), mask_lat2.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(
    mask_lon2.shape[0], mask_lon2.shape[1])
masked = np.ones_like(mask_lon2)
masked[mask] = np.nan


wind_earth = xr.open_dataset(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m201008.nc')
wind_earth_1h_100m = xr.open_dataset('scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc')

lon = wind_earth_1h_100m.lon.values
lat = wind_earth_1h_100m.lat.values
time = wind_earth_1h_100m.time.values

windlevel = np.arange(0, 18.1, 0.1)
ticks = np.arange(0, 18.1, 3)

istart = 5156  # 5156
ifinal = 5288  # 5288

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)

ims = []

for i in np.arange(istart, ifinal):
    wind = wind_earth_1h_100m.strength[i, :, :].values
    plt_wind = ax.pcolormesh(
        lon, lat, wind, cmap=cm.get_cmap('viridis', len(windlevel)),
        norm=BoundaryNorm(windlevel, ncolors=len(windlevel), clip=False),
        transform=transform, rasterized=True, zorder=-2,)
    
    iarrow = 30
    # iarrow_shift = int(iarrow/2)
    iarrow_shift = 25
    windarrow = ax.quiver(
        lon[iarrow_shift::iarrow, iarrow_shift::iarrow],
        lat[iarrow_shift::iarrow, iarrow_shift::iarrow],
        wind_earth.u_earth[i - 5088, (80 + iarrow_shift):920:iarrow,
                           (80 + iarrow_shift):920:iarrow] /
        (wind[iarrow_shift::iarrow, iarrow_shift::iarrow]),
        wind_earth.v_earth[i - 5088, (80 + iarrow_shift):920:iarrow,
                           (80 + iarrow_shift):920:iarrow] /
        (wind[iarrow_shift::iarrow, iarrow_shift::iarrow]),
        rasterized=True, zorder=-2,)
    
    wind_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC',)
    
    topomask = ax.contourf(
        lon, lat, model_topo_mask,
        colors='white', levels=np.array([0.5, 1.5]))
    topomask.__class__ = mpl.contour.QuadContourSet
    add_arts = topomask.collections
    outsidemask = ax.contourf(
        mask_lon2, mask_lat2, masked,
        colors='white', levels=np.array([0.5, 1.5]))
    outsidemask.__class__ = mpl.contour.QuadContourSet
    add_arts1 = outsidemask.collections
    
    ims.append([plt_wind, windarrow, wind_time] + add_arts + add_arts1)
    print(str(i) + '/' + str(ifinal - 1))

cbar = fig.colorbar(
    plt_wind, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=0.9, aspect=25, ticks=ticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("100-meter wind velocity [$m\;s^{-1}$]")

gl = ax.gridlines(crs=transform, linewidth=0.25, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.add_feature(borders, lw = 0.25); ax.add_feature(coastline, lw = 0.25)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

ax.set_extent(extent1km_lb, crs=transform)
scale_bar(ax, bars=2, length=200, location=(0.02, 0.015),
          barheight=20, linewidth=0.15, col='black', middle_label=False)
ax.set_rasterization_zorder(-1)

fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
ani.save(
    'figures/03_wind/3.4.1_Wind_strength_direction_in_2010080320_2010080907.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


'''
'''

# endregion
#  =============================================================================


# =============================================================================
# region wind rose in 20100803-09
# https://community.plotly.com/t/wind-rose-with-wind-speed-m-s-and-direction-deg-data-columns-need-help/33274

mpl.rc('font', family='Times New Roman', size=10)

wind_earth_1h_100m = xr.open_dataset('scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_2010.nc',
                                     chunks={"time": 5})

istart = 5156  # 5156
ifinal = 5288  # 5288

wind_strength = wind_earth_1h_100m.strength[
    istart:ifinal, :, :].values.flatten()
wind_direction = wind_earth_1h_100m.direction[
    istart:ifinal, :, :].values.flatten()

windbins = np.arange(0, 14.1, 2, dtype = 'int32')

fig, ax = plt.subplots(figsize=np.array([8.8, 7]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ax.set_extent([0, 1, 0, 1])
windrose_ax = inset_axes(
    ax, width=2.2, height=2.2, loc=10, bbox_to_anchor = (0.45, 0.5),
    bbox_transform=ax.transData, axes_class=WindroseAxes
)
windrose_ax.bar(
    wind_direction, wind_strength, normed=True,
    opening=1, edgecolor=None, nsector=36,
    bins=windbins,
    cmap=cm.get_cmap('RdBu_r', len(windbins)),
    label='Wind velocity [m/s]',
    )
windrose_legend = windrose_ax.legend(
    # labels = 'Wind velocity [m/s]',
    loc=(1.05, 0.15),
    decimal_places=0, ncol = 1,
    borderpad=0.1,
    labelspacing=0.5, handlelength=1.2, handletextpad = 0.6,
    fancybox=False,
    fontsize=8,
    frameon=False,
    title='Wind velocity [m/s]', title_fontsize = 8,
    labels=['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b'],
)
windrose_ax.grid(alpha = 0.5, ls = '--', lw = 0.5)

for lh in windrose_legend.legendHandles:
    lh.set_edgecolor(None)

windrose_ax.tick_params(axis='x', which='major', pad=0)

windrose_ax.set_yticks(np.arange(5, 25.1, step=5))
windrose_ax.set_yticklabels([5, 10, 15, 20, '25%'])

ax.axis('off')
# ax.text(0.1, 0.1, 'Mean wind direction: 79° + 180°')
fig.subplots_adjust(left=0.01, right=0.8, bottom=0.2, top=0.8)
plt.savefig('figures/03_wind/3.5.1 wind rose in 20100803_09.png', dpi=1200)
# fig.savefig('figures/00_test/trial1.png', dpi=300)


'''
270 - (np.mean(wind_direction) - 180) - 180

fig, ax = framework_plot(
    "1km_lb", figsize=np.array([8.8, 8.8]) / 2.54,
)

windrose_ax = inset_axes(
    ax, width=1, height=1, loc=10, bbox_to_anchor = (-18, 30),
    bbox_transform=ax.transData, axes_class=WindroseAxes
)
windrose_ax.bar(wind_direction, wind_strength, normed=True,
       opening=1, edgecolor='white', nsector=36,
       bins=windbins,
       cmap=cm.get_cmap('RdYlBu_r', len(windbins))
       )
windrose_ax.legend(
    loc='lower left', decimal_places=0,
    # prop={'size': 12},
    borderpad=0.1,
    labelspacing=0.002, handlelength=1,
    fontsize=10,
    frameon=False,
)

scale_bar(ax, bars=2, length=200, location=(0.04, 0.03),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          )
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
plt.savefig('figures/00_test/trial.png', dpi=600)
'''

# endregion
# =============================================================================


# =============================================================================
# region wind rose in 2010
windbins = np.arange(0, 14.1, 2, dtype='int32')
nrow = 3
ncol = 4
mpl.rc('font', family='Times New Roman', size=8)

for i in np.arange(4, 10, 1): # range(10):
    # i = 0
    # years[i]
    wind_earth_1h_100m = xr.open_dataset(
        'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_20' + years[i] + '.nc',
        chunks={"time": 5})
    time = pd.to_datetime(wind_earth_1h_100m.time.values)
    
    fig = plt.figure(figsize=np.array([6*ncol, 5*nrow]) / 2.54)
    gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
    axs = gs.subplots(subplot_kw={'projection': transform})
    
    for j in np.arange(0, nrow):
        for k in np.arange(0, ncol):
            # if True:  # k == ncol - 1 or j == nrow - 1:  # k == 0 and j == 0:  #
            # wind_strength = wind_earth_1h_100m.strength[
            #     np.vstack((time.month == j*ncol + k + 1,
            #                       time.day < 2)).all(axis=0),
            #      :, :].values.flatten()
            # wind_direction = wind_earth_1h_100m.direction[
            #     np.vstack((time.month == j*ncol + k + 1,
            #                time.day < 2)).all(axis=0),
            #     :, :].values.flatten()
            
            wind_strength = wind_earth_1h_100m.strength[
                time.month == j*ncol + k + 1, :, :].values.flatten()
            wind_direction = wind_earth_1h_100m.direction[
                time.month == j*ncol + k + 1, :, :].values.flatten()
            
            axs[j, k].set_extent([0, 1, 0, 1])
            axs[j, k].axis('off')
            axs[j, k].text(0, 0.95, month[j*ncol + k],
                           fontsize = 10, fontweight='bold')
            
            windrose_ax = inset_axes(
                axs[j, k], width=1.6, height=1.6, loc=10,
                bbox_to_anchor=(0.5, 0.5),
                bbox_transform=axs[j, k].transData, axes_class=WindroseAxes
            )
            windrose_ax.bar(
                wind_direction, wind_strength, normed=True,
                opening=1, edgecolor=None, nsector=72,
                bins=windbins,
                cmap=cm.get_cmap('RdBu_r', len(windbins)),
            )
            
            windrose_ax.tick_params(axis='x', which='major', pad=-2)
            windrose_ax.grid(alpha = 0.5, ls = '--', lw = 0.5)
            # windrose_ax.set_yticks(np.arange(5, 25.1, step=5))
            # windrose_ax.set_yticklabels([5, 10, 15, 20, '25%'])
            windrose_ax.set_yticks(np.arange(2, 12.1, step=2))
            windrose_ax.set_yticklabels([2, 4, 6, 8, 10, '12%'])
    
    # windrose_ax._info['bins'][-1] = 16
    windrose_legend = windrose_ax.legend(
        loc=(1.2, 1.2),
        decimal_places=0, ncol=1,
        borderpad=0.1,
        labelspacing=0.5,
        handlelength=1.2,
        handletextpad=0.6,
        fancybox=False,
        fontsize=10,
        frameon=False,
        title='Wind velocity [m/s]', title_fontsize=10,
        handler_map={'edgecolor': 'white'}
    )
    for lh in windrose_legend.legendHandles:
        lh.set_edgecolor(None)
    
    fig.subplots_adjust(left=0.01, right=0.85, bottom=0.01, top=0.99)
    fig.savefig(
        'figures/03_wind/03_00 annual monthly windrose/' + '3_0.' + str(i) + \
        ' wind rose in 20' + str(years[i]) + '.png', dpi=600)
    # fig.savefig('figures/03_wind/3.5.2 wind rose in 2010.png', dpi=1200)
    plt.close('all')
    print(i)



'''
'''

# endregion
# =============================================================================


# =============================================================================
# region wind rose from 2006 to 2015

windbins = np.arange(0, 14.1, 2, dtype='int32')
nrow = 3
ncol = 4
mpl.rc('font', family='Times New Roman', size=8)

filelist = sorted(glob.glob(
    '/project/pr94/qgao/DEoAI/scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_20??.nc'))

fig = plt.figure(figsize=np.array([6*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform})

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        # j=0
        # k=0
        wind_earth_1h_100m = xr.open_mfdataset(
            filelist, concat_dim="time",
            data_vars='minimal', coords='minimal', compat='override',
            chunks={'time': 1},
        )
        time = pd.to_datetime(wind_earth_1h_100m.time.values)
        
        if True:  # k == 2 and j == 1:  # k == ncol - 1 or j == nrow - 1:  #
            # wind_strength = wind_earth_1h_100m.strength[
            #     np.vstack((time.month == j*ncol + k + 1,
            #                       time.day < 2)).all(axis=0),
            #      :, :].values.flatten()
            # wind_direction = wind_earth_1h_100m.direction[
            #     np.vstack((time.month == j*ncol + k + 1,
            #                time.day < 2)).all(axis=0),
            #     :, :].values.flatten()
            
            
            wind_direction = wind_earth_1h_100m.direction[
                time.month == j*ncol + k + 1, :, :
            ].astype('int16').data.flatten()
            # wind_direction.compute()
            
            wind_strength = wind_earth_1h_100m.strength[
                time.month == j*ncol + k + 1, :, :
                    ].astype('float16').data.flatten()
            # wind_strength.compute()
            
            axs[j, k].set_extent([0, 1, 0, 1])
            axs[j, k].axis('off')
            axs[j, k].text(0, 0.95, month[j*ncol + k],
                           fontsize = 10, fontweight='bold')
            
            windrose_ax = inset_axes(
                axs[j, k], width=1.6, height=1.6, loc=10,
                bbox_to_anchor=(0.5, 0.5),
                bbox_transform=axs[j, k].transData, axes_class=WindroseAxes
            )
            windrose_ax.bar(
                wind_direction, wind_strength, normed=True,
                opening=1, edgecolor=None, nsector=72,
                bins=windbins,
                cmap=cm.get_cmap('RdBu_r', len(windbins)),
            )
            
            windrose_ax.tick_params(axis='x', which='major', pad=-2)
            windrose_ax.grid(alpha = 0.5, ls = '--', lw = 0.5)
            windrose_ax.set_yticks(np.arange(2, 12.1, step=2))
            windrose_ax.set_yticklabels([2, 4, 6, 8, 10, '12%'])
        
        print(str(j) + '/' + str(k))
        del wind_earth_1h_100m, time, wind_strength, wind_direction

windrose_legend = windrose_ax.legend(
    loc=(1.2, 1.2),
    decimal_places=0, ncol=1,
    borderpad=0.1,
    labelspacing=0.5,
    handlelength=1.2,
    handletextpad=0.6,
    fancybox=False,
    fontsize=10,
    frameon=False,
    title='Wind velocity [m/s]', title_fontsize=10,
    handler_map={'edgecolor': 'white'}
    )
for lh in windrose_legend.legendHandles:
    lh.set_edgecolor(None)

fig.subplots_adjust(left=0.01, right=0.85, bottom=0.01, top=0.99)
fig.savefig('figures/03_wind/3.8.0 wind rose from 2006 to 2015.png', dpi=1200)
plt.close('all')


'''
'''

# endregion
# =============================================================================


# =============================================================================
# region windrose in ERA5 from 2006-2015

windbins = np.arange(0, 14.1, 2, dtype='int32')
nrow = 3
ncol = 4
mpl.rc('font', family='Times New Roman', size=8)

wind100m_u = xr.open_dataset('scratch/obs/era5/hourly_100m_wind_u.nc')
wind100m_v = xr.open_dataset('scratch/obs/era5/hourly_100m_wind_v.nc')

uwind = wind100m_u.u100.values
vwind = wind100m_v.v100.values
time = pd.to_datetime(wind100m_u.time.values)

direction = mpcalc.wind_direction(
    u=uwind * units('m/s'), v=vwind * units('m/s'), convention='to').magnitude
strength = (uwind**2 + vwind**2)**0.5


fig = plt.figure(figsize=np.array([6*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform})

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        # j=0; k=0
        if True:  # k == 2 and j == 1:  # k == ncol - 1 or j == nrow - 1:  #
            
            wind_direction = direction[
                time.month == j*ncol + k + 1, :, :
            ].flatten()
            
            wind_strength = strength[
                time.month == j*ncol + k + 1, :, :
                    ].flatten()
            
            axs[j, k].set_extent([0, 1, 0, 1])
            axs[j, k].axis('off')
            axs[j, k].text(0, 0.95, month[j*ncol + k],
                           fontsize = 10, fontweight='bold')
            
            windrose_ax = inset_axes(
                axs[j, k], width=1.6, height=1.6, loc=10,
                bbox_to_anchor=(0.5, 0.5),
                bbox_transform=axs[j, k].transData, axes_class=WindroseAxes
            )
            windrose_ax.bar(
                wind_direction, wind_strength, normed=True,
                opening=1, edgecolor=None, nsector=72,
                bins=windbins,
                cmap=cm.get_cmap('RdBu_r', len(windbins)),
            )
            
            windrose_ax.tick_params(axis='x', which='major', pad=-2)
            windrose_ax.grid(alpha = 0.5, ls = '--', lw = 0.5)
            windrose_ax.set_yticks(np.arange(2, 12.1, step=2))
            windrose_ax.set_yticklabels([2, 4, 6, 8, 10, '12%'])
        
        print(str(j) + '/' + str(k))

windrose_legend = windrose_ax.legend(
    loc=(1.2, 1.2),
    decimal_places=0, ncol=1,
    borderpad=0.1,
    labelspacing=0.5,
    handlelength=1.2,
    handletextpad=0.6,
    fancybox=False,
    fontsize=10,
    frameon=False,
    title='Wind velocity $[ms^{-1}]$', title_fontsize=10,
    handler_map={'edgecolor': 'white'}
    )
for lh in windrose_legend.legendHandles:
    lh.set_edgecolor(None)

fig.subplots_adjust(left=0.01, right=0.85, bottom=0.01, top=0.99)
fig.savefig(
    'figures/03_wind/3.8.1 wind rose in ERA5 from 2006 to 2015.png',
    dpi=1200)
plt.close('all')


# endregion
# =============================================================================


# =============================================================================
# region windrose in ERA5 over the analysis region from 2006-2015

windbins = np.arange(0, 14.1, 2, dtype='int32')
nrow = 3
ncol = 4
mpl.rc('font', family='Times New Roman', size=8)

wind100m_u = xr.open_dataset('scratch/obs/era5/hourly_100m_wind_u.nc')
wind100m_v = xr.open_dataset('scratch/obs/era5/hourly_100m_wind_v.nc')

uwind = wind100m_u.u100.values
vwind = wind100m_v.v100.values
time = pd.to_datetime(wind100m_u.time.values)
pre_lon = wind100m_u.longitude.values
pre_lat = wind100m_u.latitude.values

direction = mpcalc.wind_direction(
    u=uwind * units('m/s'), v=vwind * units('m/s'), convention='to').magnitude
strength = (uwind**2 + vwind**2)**0.5

# create a mask
from matplotlib.path import Path
polygon = [
    (-23.401758, 31.897812), (-19.662523, 24.182158),
    (-11.290954, 26.82321), (-14.128447, 34.85296)
]
poly_path = Path(polygon)
x, y = np.meshgrid(pre_lon, pre_lat)
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
analysis_region_mask = poly_path.contains_points(coors).reshape(x.shape)

direction = np.ma.array(direction)
direction[:, analysis_region_mask == False] = np.ma.masked
strength = np.ma.array(strength)
strength[:, analysis_region_mask == False] = np.ma.masked


fig = plt.figure(figsize=np.array([6*ncol, 5*nrow]) / 2.54)
gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.01)
axs = gs.subplots(subplot_kw={'projection': transform})

for j in np.arange(0, nrow):
    for k in np.arange(0, ncol):
        # j=0; k=0
        if True:  # k == 2 and j == 1:  # k == ncol - 1 or j == nrow - 1:  #
            
            wind_direction = direction[
                time.month == j*ncol + k + 1, :, :
            ].flatten()
            
            wind_strength = strength[
                time.month == j*ncol + k + 1, :, :
            ].flatten()
            
            axs[j, k].set_extent([0, 1, 0, 1])
            axs[j, k].axis('off')
            axs[j, k].text(0, 0.95, month[j*ncol + k],
                           fontsize = 10, fontweight='bold')
            
            windrose_ax = inset_axes(
                axs[j, k], width=1.6, height=1.6, loc=10,
                bbox_to_anchor=(0.5, 0.5),
                bbox_transform=axs[j, k].transData, axes_class=WindroseAxes
            )
            windrose_ax.bar(
                wind_direction[wind_direction.mask == False],
                wind_strength[wind_strength.mask == False],
                normed=True,
                opening=1, edgecolor=None, nsector=72,
                bins=windbins,
                cmap=cm.get_cmap('RdBu_r', len(windbins)),
            )
            
            windrose_ax.tick_params(axis='x', which='major', pad=-2)
            windrose_ax.grid(alpha = 0.5, ls = '--', lw = 0.5)
            windrose_ax.set_yticks(np.arange(2, 12.1, step=2))
            windrose_ax.set_yticklabels([2, 4, 6, 8, 10, '12%'])
        
        print(str(j) + '/' + str(k))

windrose_legend = windrose_ax.legend(
    loc=(1.2, 1.2),
    decimal_places=0, ncol=1,
    borderpad=0.1,
    labelspacing=0.5,
    handlelength=1.2,
    handletextpad=0.6,
    fancybox=False,
    fontsize=10,
    frameon=False,
    title='Wind velocity $[m \; s^{-1}]$', title_fontsize=10,
    handler_map={'edgecolor': 'white'}
    )
for lh in windrose_legend.legendHandles:
    lh.set_edgecolor(None)

fig.subplots_adjust(left=0.01, right=0.85, bottom=0.01, top=0.99)
fig.savefig(
    'figures/03_wind/3.8.2 wind rose in ERA5 over analysis region from 2006 to 2015.png',
    dpi=1200)
plt.close('all')


# endregion
# =============================================================================

