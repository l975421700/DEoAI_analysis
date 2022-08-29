

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
# region domain large vorticity

folder_rvorticity_1km_1h_100m = \
    'scratch/rvorticity/relative_vorticity_1km_1h_100m'

filelist_rvorticity_1km_1h_100m = \
    sorted(glob.glob(folder_rvorticity_1km_1h_100m + '/*100m2010*'))

rvorticity_1km_1h_100m = xr.open_mfdataset(
    filelist_rvorticity_1km_1h_100m, concat_dim="time",
    data_vars='minimal', coords='minimal', compat='override'
)

rvorticity_1km_1h_100m_average = xr.open_dataset(
    "scratch/rvorticity/rvorticity_1km_1h_100m_average.nc"
)

rlon = rvorticity_1km_1h_100m.rlon[80:920].data
rlat = rvorticity_1km_1h_100m.rlat[80:920].data
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].data
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].data

# len(np.where(rvorticity_1km_1h_100m_average.m_abs.values > 0.0003)[0] )

time = rvorticity_1km_1h_100m.time[
    np.where(rvorticity_1km_1h_100m_average.m_abs.values > 0.0003)[0]
]
rvorticity_1km_1h_100m_mean_0003 = xr.Dataset(
    {"relative_vorticity": (
        ("time", "rlat", "rlon"),
        np.zeros((len(time), len(rlat), len(rlon)))
    ),
        "lat": (("rlat", "rlon"), lat),
        "lon": (("rlat", "rlon"), lon),
    },
    coords={
        "time": time,
        "rlat": rlat,
        "rlon": rlon,
    }
)

rvorticity_1km_1h_100m_mean_0003.relative_vorticity[:, :, :] = \
    rvorticity_1km_1h_100m.relative_vorticity[
        np.where(rvorticity_1km_1h_100m_average.m_abs.values > 0.0003)[0],
        80:920, 80:920
    ]

rvorticity_1km_1h_100m_mean_0003.to_netcdf(
    'scratch/rvorticity/rvorticity_1km_1h_100m_mean_0003.nc'
)


# endregion
# =============================================================================


# =============================================================================
# region domain large vorticity gaussian_filter, nope!!!


# rvorticity_1km_1h_100m_mean_0003 = xr.open_dataset(
#     'scratch/rvorticity/exp_2010/rvorticity_1km_1h_100m_mean_0003.nc'
# )

# rvorticity_1km_1h_100m_mean_0003_smoothed = rvorticity_1km_1h_100m_mean_0003
# import scipy.ndimage as ndimage
# rvorticity_1km_1h_100m_mean_0003_smoothed.relative_vorticity[:, :, :] = \
#     ndimage.gaussian_filter(
#         rvorticity_1km_1h_100m_mean_0003.relative_vorticity[:, :, :],
#         sigma=1, order=0)

# rvorticity_1km_1h_100m_mean_0003_smoothed.to_netcdf(
#     'scratch/rvorticity/exp_2010/rvorticity_1km_1h_100m_mean_0003_smoothed.nc'
# )

# endregion
# =============================================================================


# =============================================================================
# region plot relative vorticity in 20100220

# load data
rvorticity_1km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201008.nc'
)
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].values
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].values
time = rvorticity_1km_1h_100m.time.values

# create a color map
top = cm.get_cmap('Blues_r', 200)
bottom = cm.get_cmap('Reds', 200)
newcolors = np.vstack((top(np.linspace(0, 1, 120)),
                       [1, 1, 1, 1],
                       bottom(np.linspace(0, 1, 120))))
newcmp = ListedColormap(newcolors, name='RedsBlues_r')

# set colormap level and ticks
vorlevel = np.arange(-12, 12.1, 0.1)
ticks = np.arange(-12, 12.1, 3)

mpl.rc('font', family='Times New Roman', size=16)

with PdfPages(
        'figures/02_vorticity/2.4.2_Relative_vorticity_in_201008.pdf') as pdf:
    # def rvorplot(file):
    #     pdf = PdfPages(file)
    nrow = 3
    ncol = 4
    shift = 0
    istart = 0
    ifinal = int(rvorticity_1km_1h_100m.time.shape[0]/ncol/nrow)
    # np.arange(0, int(rvorticity_1km_1h_100m.time.shape[0]/ncol/nrow))
    for i in np.arange(istart, ifinal):
        
        fig = plt.figure(figsize=np.array([8*ncol, 8.8*nrow]) / 2.54)
        gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.05)
        axs = gs.subplots(subplot_kw={'projection': transform})
        
        for j in np.arange(0, nrow):
            for k in np.arange(0, ncol):
                if True:  # i == istart and k == 0 and j == 0:  #
                    rvor = rvorticity_1km_1h_100m.relative_vorticity[
                        i*nrow*ncol + j*ncol + k + shift,
                        80:920, 80:920].values * 10**4
                    
                    plt_rvor = axs[j, k].pcolormesh(
                        lon, lat, rvor, cmap=newcmp,
                        norm=BoundaryNorm(
                            vorlevel, ncolors=newcmp.N, clip=False),
                        transform=transform, zorder=-2, rasterized=True)
                
                # add borders, gridlines, title and set extent
                axs[j, k].add_feature(coastline)
                axs[j, k].add_feature(borders)
                
                gl = axs[j, k].gridlines(
                    crs=transform, linewidth=0.5,
                    color='gray', alpha=0.5, linestyle='--')
                gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
                gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
                
                axs[j, k].set_title(
                    str(time[i*nrow*ncol+j*ncol+k+shift])[11:13] + ':00:00')
                
                axs[j, k].set_extent(extent1km_lb, crs=transform)
                
                axs[j, k].set_rasterization_zorder(-1)
                print(str(k) + '/' + str(ncol-1) + '  ' +
                    str(j) + '/' + str(nrow-1) + '  ' + \
                        str(i) + '/' + str(ifinal-1))
        
        cbar = fig.colorbar(
            plt_rvor, ax = axs, orientation="horizontal",  pad=0.1,
            fraction=0.09, shrink=0.6, aspect=25, anchor = (0.5, -0.8),
            ticks=ticks, extend='both')
        cbar.ax.set_xlabel(
            "Relative vorticity [$10^{-4}\;s^{-1}$]  at  " + \
                str(time[i*nrow*ncol])[0:10])
        
        # fig.tight_layout()
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.99)
        pdf.savefig(fig, dpi=200)
        plt.close('all')


'''
# check runing time
from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(rvorplot)
lp_wrapper(file='figures/02_vorticity/2.4.2_Relative_vorticity_in_201002.pdf')
lp.print_stats()

%prun rvorplot('figures/02_vorticity/2.4.2_Relative_vorticity_in_201002.pdf')


'''



# endregion
# =============================================================================


# =============================================================================
# region plot relative vorticity with animation in 20100803-09

# load data
rvorticity_1km_1h_100m = xr.open_dataset(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m201008.nc'
)
lon = rvorticity_1km_1h_100m.lon[80:920, 80:920].values
lat = rvorticity_1km_1h_100m.lat[80:920, 80:920].values
time = rvorticity_1km_1h_100m.time.values


# set colormap level and ticks
vorlevel = np.arange(-12, 12.1, 0.1)
ticks = np.arange(-12, 12.1, 3)

# create a color map
top = cm.get_cmap('Blues_r', int(np.floor(len(vorlevel) / 2)))
bottom = cm.get_cmap('Reds', int(np.floor(len(vorlevel) / 2)))
newcolors = np.vstack(
    (top(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2)))),
     [1, 1, 1, 1],
     bottom(np.linspace(0, 1, int(np.floor(len(vorlevel) / 2))))))
newcmp = ListedColormap(newcolors, name='RedsBlues_r')


# https://stackoverflow.com/questions/61652069/matplotlib-artistanimation-plot-entire-figure-in-each-step

fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.8]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ims = []
istart = 68
ifinal = 200 # 200

for i in np.arange(istart, ifinal):
    rvor = rvorticity_1km_1h_100m.relative_vorticity[
        i, 80:920, 80:920].values * 10**4
    plt_rvor = ax.pcolormesh(
        lon, lat, rvor, cmap=newcmp, rasterized=True, transform=transform,
        norm=BoundaryNorm(vorlevel, ncolors=newcmp.N, clip=False), zorder=-2,)
    
    rvor_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC')
    
    ims.append([plt_rvor, rvor_time])
    print(str(i) + '/' + str(ifinal - 1))

gl = ax.gridlines(crs=transform, linewidth=0.25, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.add_feature(borders, lw = 0.25); ax.add_feature(coastline, lw = 0.25)
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])

cbar = fig.colorbar(
    plt_rvor, ax=ax, orientation="horizontal",  pad=0.1, fraction=0.09,
    shrink=1, aspect=25, ticks=ticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Relative vorticity [$10^{-4}\;s^{-1}$]")

scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False)
ax.set_extent(extent1km_lb, crs=transform)
ax.set_rasterization_zorder(-1)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.08, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)

ani.save(
    'figures/02_vorticity/2.4.4_Relative_vorticity_in_2010080320_2010080907_150.mp4',
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

# https://unidata.github.io/python-training/workshop/Satellite_Data/satellite-animations/
# from IPython.display import HTML
# HTML(anim.to_jshtml())


# endregion
# =============================================================================


# =============================================================================
# region plot precipitation in 20100220

mpl.rc('font', family='Times New Roman', size=16)

filelist_1h_TOT_PREC = sorted(glob.glob(
    folder_1km + '1h_TOT_PREC/lffd2010080*[0-9].nc'))

TOT_PREC_1h = xr.open_mfdataset(
    filelist_1h_TOT_PREC, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override')

lon = TOT_PREC_1h.lon[80:920, 80:920].values
lat = TOT_PREC_1h.lat[80:920, 80:920].values
time = TOT_PREC_1h.time.values

tprange = [0.1, 1, 5, 10, 20, 50, 100]
tplabels = ['0.1', '1', '5', '10', '20', '50', '100']
norm = mpcolors.LogNorm(vmin=0.1, vmax=100.)
cmap = plt.cm.Blues
color_bg = plt.cm.Greys(0.25)

preclevel = np.arange(0, 50.1, 0.5)
ticks = np.arange(0, 50.1, 5)


with PdfPages(
        'figures/04_precipitation/4.0.0_precipitation_in_2010080.pdf') as pdf:
    nrow = 3
    ncol = 4
    shift = 1
    istart = 0
    ifinal = 1  # int(time.shape[0]/ncol/nrow)  #
    
    for i in np.arange(istart, ifinal):
        fig = plt.figure(figsize=np.array([8*ncol, 8.8*nrow]) / 2.54)
        gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.05)
        axs = gs.subplots(subplot_kw={'projection': transform})
        for j in np.arange(0, nrow):
            for k in np.arange(0, ncol):
                if True:  # i == istart and k == 0 and j == 0:  #
                    prec = TOT_PREC_1h.TOT_PREC[
                        i*nrow*ncol + j*ncol + k + shift,
                        80:920, 80:920].values
                    
                    plt_prec = axs[j, k].pcolormesh(
                        lon, lat, prec, cmap=cmap,
                        norm=mpcolors.LogNorm(vmin=0.1, vmax=100.),
                        transform=transform, zorder=-2, rasterized=True)
                
                # add borders, gridlines, title and set extent
                axs[j, k].add_feature(coastline)
                axs[j, k].add_feature(borders)
                axs[j, k].set_facecolor(color_bg)
                domain = axs[j, k].fill(lon, lat, "white",
                                 transform=transform, zorder=-10)
                
                gl = axs[j, k].gridlines(
                    crs=transform, linewidth=0.5,
                    color='gray', alpha=0.5, linestyle='--')
                gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
                gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
                
                axs[j, k].set_title(
                    str(time[i*nrow*ncol+j*ncol+k+shift])[11:13] + ':00:00')
                
                axs[j, k].set_extent(extent1km_lb, crs=transform)
                
                axs[j, k].set_rasterization_zorder(-1)
                print(str(k) + '/' + str(ncol-1) + '  ' +
                    str(j) + '/' + str(nrow-1) + '  ' + \
                        str(i) + '/' + str(ifinal-1))
        cbar = fig.colorbar(
            plt_prec, ax=axs, orientation="horizontal",  pad=0.1,
            fraction=0.09, shrink=0.6, aspect=25, anchor = (0.5, -0.8),
            ticks=tprange, extend='max')
        cbar.ax.set_xticklabels(tplabels)
        cbar.ax.set_xlabel(
            "Precipitation [mm/h]  at  " + str(time[i*nrow*ncol])[0:10])
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.99)
        pdf.savefig(fig, dpi=200)
        plt.close('all')


'''
ddd = TOT_PREC_1h.TOT_PREC.values[TOT_PREC_1h.TOT_PREC.values > 0.1]
plt.hist(ddd, bins=60)
plt.savefig('figures/02_vorticity/test.png')
plt.close('all')

np.percentile(a=ddd, q=quantiles[0])

'''
# endregion
# =============================================================================







