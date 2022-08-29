


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
# region cloud plot

mpl.rc('font', family='Times New Roman', size=16)

filelist_1h = sorted(glob.glob(folder_1km + '1h/lffd20100220*[0-9].nc'))

clouds_1h = xr.open_mfdataset(
    filelist_1h, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override')

lon = clouds_1h.lon[80:920, 80:920].values
lat = clouds_1h.lat[80:920, 80:920].values
time = clouds_1h.time.values

# Transparent colormap
colors = [(1,1,1,c) for c in np.linspace(0,1,100)]
cmapwhite = mpcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
color_bg = plt.cm.Blues(0.25)
vmax = 4

with PdfPages(
        'figures/05_clouds/5.0.0_Clouds_in_20100220.pdf') as pdf:
    # def rvorplot(file):
    #     pdf = PdfPages(file)
    nrow = 3
    ncol = 4
    shift = 0
    istart = 0
    ifinal = int(time.shape[0]/ncol/nrow)  # 1  #
    for i in np.arange(istart, ifinal):
        
        fig = plt.figure(figsize=np.array([8*ncol, 8.8*nrow]) / 2.54)
        gs = fig.add_gridspec(nrow, ncol, hspace=0.01, wspace=0.05)
        axs = gs.subplots(subplot_kw={'projection': transform})
        
        for j in np.arange(0, nrow):
            for k in np.arange(0, ncol):
                if True:  # i == istart and k == 0 and j == 0:  #
                    tqc = clouds_1h.TQC[
                        i*nrow*ncol + j*ncol + k + shift,
                        80:920, 80:920].values
                    
                    plt_tqc = axs[j, k].pcolormesh(
                        lon, lat, tqc, cmap=cmapwhite, vmin=0.0, vmax=vmax,
                        transform=transform, zorder=-2, rasterized=True)
                    
                    # add borders, gridlines, title and set extent
                    # axs[j, k].background_img(
                    #     name='blue_marble_jun', resolution='high')
                    # axs[j, k].background_img(
                    #     name='natural_earth', resolution='low')
                
                # add borders, gridlines, title and set extent
                axs[j, k].add_feature(coastline)
                axs[j, k].add_feature(borders)
                axs[j, k].set_facecolor(color_bg)
                
                axs[j, k].set_title(
                    str(time[i*nrow*ncol+j*ncol+k+shift])[11:13] + ':00:00')
                
                axs[j, k].set_extent(extent1km_lb, crs=transform)
                
                axs[j, k].set_rasterization_zorder(-1)
                print(str(k) + '/' + str(ncol-1) + '  ' +
                    str(j) + '/' + str(nrow-1) + '  ' + \
                        str(i) + '/' + str(ifinal-1))
        
        # cbar = fig.colorbar(
        #     plt_tqc, ax=axs, orientation="horizontal", pad=0.1, extend='both',
        #     fraction=0.09, shrink=0.6, aspect=25, anchor = (0.5, -0.8),
        #     # ticks=ticks,
        #     )
        # cbar.ax.set_xlabel(
        #     "Relative vorticity [$10^{-4}\;s^{-1}$]  at  " + \
        #         str(time[i*nrow*ncol])[0:10])
        
        fig.text(0.5, 0.04, 'Clouds at ' + str(time[i*nrow*ncol])[0:10],
                 ha = 'center', va = 'center')
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.99)
        pdf.savefig(fig, dpi=200)
        plt.close('all')


'''
# check runing time
from line_profiler import LineProfiler
lp = LineProfiler()
lp_wrapper = lp(rvorplot)
lp_wrapper(file='figures/05_clouds/5.0.0_Clouds_in_20100220.pdf')
lp.print_stats()

# set colormap level and ticks
vorlevel = np.arange(-12, 12.1, 0.1)
ticks = np.arange(-12, 12.1, 3)


'''


# endregion
# =============================================================================


# =============================================================================
# region cloud animation

filelist_1h = sorted(glob.glob(folder_1km + '1h/lffd2010080*[0-9].nc'))

clouds_1h = xr.open_mfdataset(
    filelist_1h, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override')

lon = clouds_1h.lon[80:920, 80:920].values
lat = clouds_1h.lat[80:920, 80:920].values
time = clouds_1h.time.values

# Transparent colormap
colors = [(1,1,1,c) for c in np.linspace(0,1,100)]
cmapwhite = mpcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
color_bg = plt.cm.Blues(0.5)
vmax = 1


fig, ax = plt.subplots(1, 1, figsize=np.array([8.8, 8.2]) / 2.54,
                       subplot_kw={'projection': transform}, dpi=600)
ims = []
istart = 68
ifinal = 200  # 200

for i in np.arange(istart, ifinal):
    
    tqc = clouds_1h.TQC[i, 80:920, 80:920].values
    
    plt_tqc = ax.pcolormesh(
        lon, lat, tqc, cmap=cmapwhite, vmin=0.0, vmax=vmax,
        transform=transform, rasterized=True,
        # zorder=-2,
        )
    
    tqc_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC',
        color = 'white',
        )
    
    ims.append([plt_tqc, tqc_time])
    print(str(i) + '/' + str(ifinal - 1))

scale_bar(ax, bars=2, length=200, location=(0.08, 0.06),
          barheight=20, linewidth=0.15, col='black', middle_label=False,
          fontcolor='white')
# ax.stock_img()
# ax.background_img(name='blue_marble_jun', resolution='high')
# ax.background_img(name='natural_earth', resolution='high')
ax.set_xticks(ticklabel1km_lb[0]); ax.set_xticklabels(ticklabel1km_lb[1])
ax.set_yticks(ticklabel1km_lb[2]); ax.set_yticklabels(ticklabel1km_lb[3])
ax.set_xlabel('Atmospheric cloud liquid water content [mm]')

ax.add_feature(coastline)
ax.add_feature(borders)
ax.set_facecolor(color_bg)

gl = ax.gridlines(crs=transform, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
gl.ylocator = mticker.FixedLocator(ticklabel1km_lb[2])

ax.set_extent(extent1km_lb, crs=transform)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.99)

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
ani.save('figures/05_clouds/5.0.1_Clouds_in_2010080320_2010080907.mp4')


'''

# https://unidata.github.io/python-training/workshop/Satellite_Data/satellite-animations/
# from IPython.display import HTML
# HTML(anim.to_jshtml())


'''
# endregion
# =============================================================================


