

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

from DEoAI_analysis.module.vortex_namelist import correctly_reidentified

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


# =============================================================================
# region import data
iyear = 4
# years[iyear]

################################ 100m vorticity
rvor_100m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
rvor_100m = xr.open_mfdataset(
    rvor_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = rvor_100m.time.values
lon = rvor_100m.lon[80:920, 80:920].values
lat = rvor_100m.lat[80:920, 80:920].values

################################ model topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values

################################ parameters setting
grid_size = 1.2  # in km^2
median_filter_size = 3
maximum_filter_size = 50

min_rvor = 3.
min_max_rvor = 4.
min_size = 100.
min_size_theta = 450.
min_size_dir = 450
min_size_dir1 = 900
# max_dir = 30
# max_dir1 = 60
# max_dir2 = 90
max_dir = 30
max_dir1 = 40
max_dir2 = 50
max_distance2radius = 5


################################ original simulation to calculate surface theta
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')
))

################################ create a mask for Madeira
polygon = [
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
]
poly_path = Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
madeira_mask = poly_path.contains_points(coors).reshape(840, 840)

################################ import wind
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

# endregion
# =============================================================================


# =============================================================================
# region plot reidentified rvor animation

######## mask topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
ifinal = np.where(time == np.datetime64('2010-08-09T08:00:00.000000000'))[0][0]
# istart = np.where(time == np.datetime64('2010-08-06T21:00:00.000000000'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-08-06T23:00:00.000000000'))[0][0]
initial = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]

inputfile = 'scratch/rvorticity/rvor_identify/re_identify/identified_transformed_rvor_20100803_09_stricter_dir.h5'
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_2.0 Re_Identified transformed rvor and local theta anomaly_20100803_09_001_detailed.mp4'
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_2.1 Re_Identified transformed rvor 20100803_09_002_formal.mp4'
outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_2.4 Re_Identified transformed rvor 20100803_09_005_formal_contour.mp4'

identified_rvor = tb.open_file(inputfile, mode="r")
experiment = identified_rvor.root.exp1

# time[istart: ifinal]
# orig_simulation_f[[istart, ifinal - 1]]

fig, ax = framework_plot1(
    "1km_lb", dpi=600, figsize=np.array([8.8, 9.3]) / 2.54,
    figure_margin={'left': 0.12, 'right': 0.99, 'bottom': 0.12, 'top': 0.99})
ims = []

for i in np.arange(istart, ifinal): # np.arange(istart + 5, istart + 6):  #
    # i = istart + 5
    ################################ import vorticity
    rvor = rvor_100m.relative_vorticity[i, 80:920, 80:920].values * 10**4
    
    ################################ plot relative vorticity
    plt_rvor = ax.pcolormesh(
        lon, lat, rvor, cmap=rvor_cmp, rasterized=True, transform=transform,
        norm=BoundaryNorm(rvor_level, ncolors=rvor_cmp.N, clip=False),
        zorder=-2,)
    
    ################################ plot identified vortices
    plt_contour = ax.contour(
        lon, lat, experiment.vortex_info.cols.is_vortex[i-initial],
        colors='lime', levels=np.array([-0.5, 0.5]),
        linewidths=0.3, linestyles='solid'
        )
    plt_contour.__class__ = mpl.contour.QuadContourSet
    add_arts = plt_contour.collections
    
    topomask = ax.contourf(
        lon, lat, model_topo_mask,
        colors='white', levels=np.array([0.5, 1.5]))
    topomask.__class__ = mpl.contour.QuadContourSet
    add_arts1 = topomask.collections
    
    ################################ extract vortices information
    is_vortex = experiment.vortex_info.cols.is_vortex[i - initial]
    vortex_indices = \
        np.ma.array(experiment.vortex_info.cols.vortex_indices[i - initial])
    vortex_indices[is_vortex == 0] = np.ma.masked
    vortex_count = experiment.vortex_info.cols.vortex_count[i - initial]
    
    vortices_plot = []
    for j in range(vortex_count):
        # j = 0
        ################################ extract individual vortex detial info
        vortex_de_r = [
            row[:] for row in experiment.vortex_de_info.where(
                '(time_index == ' + str(i) +
                ') & (vortex_index == ' + str(j) + ')')][0]
        vortex_de_cols = experiment.vortex_de_info.colnames
        center_lat = vortex_de_r[vortex_de_cols.index('center_lat')]
        center_lon = vortex_de_r[vortex_de_cols.index('center_lon')]
        vortex_radius = vortex_de_r[vortex_de_cols.index('radius')]
        vortex_wind_u = vortex_de_r[vortex_de_cols.index('mean_wind_u')]
        vortex_wind_v = vortex_de_r[vortex_de_cols.index('mean_wind_v')]
        vortex_size = vortex_de_r[vortex_de_cols.index('size')]
        vortex_angle = vortex_de_r[vortex_de_cols.index('angle')]
        distance2radius = vortex_de_r[vortex_de_cols.index('distance2radius')]
        ################################ plot vortex mean winds
        # plt_wind = ax.quiver(
        #     center_lon, center_lat, vortex_wind_u, vortex_wind_v,
        #     rasterized=True)
        
        ################################ plot a nominal circle
        # vortex_circle = plt.Circle(
        #     (center_lon, center_lat), vortex_radius/1.1*0.01,
        #     edgecolor='lime', facecolor='None', lw=0.3, zorder=2)
        # plt_circle = ax.add_artist(vortex_circle)
        
        ################################ plot text
        # if (j in correctly_reidentified[str(time[i])[0:13]]):
        #     text_color = 'm'
        # else:
        #     text_color = 'lime'
        # plt_text = ax.text(
        #     center_lon, center_lat, str(j) + ':' + str(int(vortex_size)) + \
        #     ':' + str(int(vortex_angle)) + ':' + \
        #     str(round(distance2radius, 1)),
        #     color=text_color, size=6, fontweight='normal')
        
        if (not j in correctly_reidentified[str(time[i])[0:13]]):
            plt_mend = ax.plot(
                center_lon + 0.4, center_lat, 'X', markersize=9,
                markerfacecolor='c', markeredgecolor='c', alpha=0.75
            )
            
            vortices_plot.append(plt_mend[0])
        # vortices_plot.append(plt_wind)
        # vortices_plot.append(plt_circle)
        # vortices_plot.append(plt_text)
    
    ################################ import theta
    # orig_simulation = xr.open_dataset(orig_simulation_f[i])
    # pres = orig_simulation.PS[0, 80:920, 80:920].values
    # tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
    # theta = tem2m * (p0sl/pres)**(r/cp)
    # theta_filtered = median_filter(theta, size=3)
    # theta_ext = maximum_filter(theta_filtered, 50, mode='nearest')
    # theta_anomalies = np.where(theta_ext == theta_filtered)
    # theta_scatter = ax.scatter(
    #     lon[theta_anomalies], lat[theta_anomalies], s=1, c='b',)
    
    ################################ add time information
    rvor_time = ax.text(
        -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC')
    
    ################################ add artists to list
    # ims.append(add_arts + [plt_rvor, rvor_time, theta_scatter] + vortices_plot)
    ims.append(add_arts + add_arts1 + [plt_rvor, rvor_time] + vortices_plot)
    print(str(i) + '/' + str(ifinal - 1))

################################ plot colorbar
cbar = fig.colorbar(
    plt_rvor, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.09,
    shrink=1, aspect=25, ticks=rvor_ticks, extend='both',
    anchor=(0.5, 1.5), panchor=(0.5, 0))
cbar.ax.set_xlabel(
    "100-meter relative vorticity [$10^{-4}\;s^{-1}$]\nIdentified vortices          Falsely identified vortices")
################################ plot legend
# ax.plot(
#     -13, 34, 'X', markersize=9, markerfacecolor='c',
#     markeredgecolor='c', alpha=0.75)
# ax.add_patch(Rectangle((-13.5, 33), 1, 0.6,
#                        ec='lime', color='None', lw=0.5))
ax.plot(
    -18.25, 20.7, 'X', markersize=8, markerfacecolor='c',
    markeredgecolor='c', alpha=0.75, clip_on=False)
ax.add_patch(Rectangle((-24, 20.5), 0.6, 0.4,
                       ec='lime', color='None', lw=0.5, clip_on=False))
################################ set rasterization
ax.set_rasterization_zorder(-10)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)


identified_rvor.close()

'''
'''
# endregion
# =============================================================================





