

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

################################ 10m vorticity
rvor_10m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h/relative_vorticity_1km_1h20' + years[iyear] + '*.nc')))
rvor_10m = xr.open_mfdataset(
    rvor_10m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

################################ model topography
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values

################################ parameters setting
min_rvor = 3.
min_max_rvor = 6.
min_size = 100
max_distance2radius = 4.5
max_ellipse_eccentricity = 3
small_size_cell = {
    'size': min_size * 1.4,
    'max_ellipse_eccentricity': max_ellipse_eccentricity / 2,
    'peak_magnitude': min_max_rvor * 1.5,
    'max_distance2radius': max_distance2radius - 1}
reject_info = False

################################ identified vortices
# identified_rvor = tb.open_file(
#     "scratch/rvorticity/rvor_identify/identified_transformed_rvor_20100803_09_002_rejected.h5",
#     mode="r")
identified_transformed_rvor_2010 = tb.open_file(
    "scratch/rvorticity/rvor_identify/decades_past/identified_transformed_rvor_2010.h5",
    mode="r")
exp2010 = identified_transformed_rvor_2010.root.exp1

################################ original simulation to calculate surface theta
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')
))

################################ create a mask for Madeira
from matplotlib.path import Path
polygon=[
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
    ]
poly_path=Path(polygon)
x, y = np.mgrid[:840, :840]
coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
mask = poly_path.contains_points(coors).reshape(840, 840)

################################ import wind
wind_100m_f = np.array(sorted(glob.glob(
    'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
wind_100m = xr.open_mfdataset(
    wind_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})

#### create a boundary
dsconst = xr.open_dataset(
    'scratch/simulation/20100801_09_3d/02_lm/lfsd20051101000000c.nc')
model_topo = dsconst.HSURF[0].values
model_topo_mask = np.ones_like(model_topo)
model_topo_mask[model_topo == 0] = np.nan

# endregion
# =============================================================================


# =============================================================================
# region define functions to find and plot local maximum

def find_plot_maxmin_points(
        lon, lat, data, ax, extrema, nsize, pointsize, pointcolor):
    """
    lon: 1D longitude
    lat: 1D latitude
    data: variables field
    ax: axis to plot
    extrema: 'min' or 'max'
    nsize: Size of the grid box to filter
    
    color: colors points
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter, maximum_filter, \
        minimum_filter, median_filter
    import numpy as np
    
    # data = gaussian_filter(data, sigma=3.0)
    data = median_filter(data, size=3)
    # add dummy variables to avoid identical neighbor values
    dummy = np.random.normal(0, 0.000001, data.shape)
    data = data + dummy
    
    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')
    
    mxy, mxx = np.where(data_ext == data)
    
    for i in range(len(mxy)):
        # 1st criterion
        # criteria1 = ((mxx[i] > 0.05 * nx) & (mxx[i] < 0.95 * nx) &
        #              (mxy[i] > 0.05 * ny) & (mxy[i] < 0.95 * ny))
        criteria1 = True
        if criteria1:
            ax.scatter(
                lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                s=pointsize, c=pointcolor,
            )
            # ax.text(
            #     lon[mxx[i]], lat[mxy[i]], symbol,
            #     color=color, clip_on=True, clip_box=ax.bbox, fontweight='bold',
            #     horizontalalignment='center', verticalalignment='center')
        # if (criteria1 & plotValue):
        #     ax.text(
        #         lon[mxx[i]], lat[mxy[i]],
        #         '\n' + str(np.int(data[mxy[i], mxx[i]])),
        #         color=color, clip_on=True, clip_box=ax.bbox,
        #         fontweight='bold', horizontalalignment='center',
        #         verticalalignment='top')

'''
lon
lat
data = theta
extrema = 'max'
nsize = 50
'''
# endregion
# =============================================================================


# =============================================================================
# region visualization 100m rvor in 201008 and 201002

istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
ifinal = np.where(time == np.datetime64('2010-08-09T08:00:00.000000000'))[0][0]
outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_0.0 Identified transformed rvor and local theta anomaly_20100803_09.pdf'
# istart = np.where(time == np.datetime64('2010-02-14T00:00:00.000000000'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-02-22T00:00:00.000000000'))[0][0]
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_0.1 Identified transformed rvor and local theta anomaly_20100214_21.pdf'
# time[istart: ifinal]
# orig_simulation_f[istart]

with PdfPages(outputfile) as pdf:
    for i in np.arange(istart, ifinal):  # np.arange(istart, istart + 4):  #
        # i = istart
        
        ################################ extract vorticity, wind
        rvor = rvor_100m.relative_vorticity[
            i, 80:920, 80:920].values * 10**4
        wind_u = wind_100m.u_earth[i, 80:920, 80:920].values
        wind_v = wind_100m.v_earth[i, 80:920, 80:920].values
        
        ################################ plot rvor
        fig, ax = framework_plot1(
            "1km_lb",
            plot_vorticity=True,
            xlabel="100m relative vorticity [$10^{-4}\;s^{-1}$]",
            vorticity_elements={
                'rvor': rvor,
                'lon': lon,
                'lat': lat,
                'vorlevel': np.arange(-12, 12.1, 0.1),
                'ticks': np.arange(-12, 12.1, 3),
                'time_point': time[i],
                'time_location': [-23, 34], },
            dpi=300
        )
        
        ################################ extract vortices information
        is_vortex = exp2010.vortex_info.cols.is_vortex[i]
        vortex_indices = np.ma.array(exp2010.vortex_info.cols.vortex_indices[i])
        vortex_indices[is_vortex == 0] = np.ma.masked
        vortex_count = exp2010.vortex_info.cols.vortex_count[i]
        for j in range(vortex_count):
            # j = 0
            ################################ extract individual vortex info
            
            #### vortex points
            vortex_points = np.where(vortex_indices == j)
            vortex_points_center = (
                vortex_points[0].mean(), vortex_points[1].mean())
            from_maderira = poly_path.contains_point(vortex_points_center)
            if from_maderira:
                
                ################################ vortex detailed info
                vortex_de_r = [row[:]
                              for row in exp2010.vortex_de_info.where(
                                  '(time_index == ' + str(i) +
                                  ') & (vortex_index == ' + str(j) + ')')][0]
                vortex_de_cols = exp2010.vortex_de_info.colnames
                center_lat = vortex_de_r[vortex_de_cols.index('center_lat')]
                center_lon = vortex_de_r[vortex_de_cols.index('center_lon')]
                vortex_radius = vortex_de_r[vortex_de_cols.index('radius')]
                
                ################################ plot vortex mean winds
                vortex_wind_u = wind_u[vortex_points].mean()
                vortex_wind_v = wind_v[vortex_points].mean()
                ax.quiver(center_lon, center_lat, vortex_wind_u, vortex_wind_v,
                          rasterized=True)
                
                ################################ calculate direction diff
                vector_1 = [vortex_wind_u, vortex_wind_v]
                vector_2 = [center_lon - center_madeira[0],
                            center_lat - center_madeira[1]]
                unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                dot_product = np.dot(unit_vector_1, unit_vector_2)
                angle = np.rad2deg(np.arccos(dot_product))
                
                ################################ plot a nominal circle
                vortex_circle = plt.Circle(
                    (center_lon, center_lat), vortex_radius/1.1*0.01,
                    edgecolor='lime', facecolor = 'None', lw = 0.3, zorder = 2)
                ax.add_artist(vortex_circle)
                
                ################################ plot text
                ax.text(
                    center_lon, center_lat,
                    str(j) + ':' + str(int(len(vortex_points[0]) * 1.2)) + \
                    ':' + str(int(angle)),
                    color='m', size=6, fontweight='normal')
            # else:
            #     is_vortex[vortex_points] = 0
        
        ################################ plot identified vortices
        ax.contour(lon, lat, is_vortex,
                   colors='lime', levels=np.array([-0.5, 0.5]),
                   linewidths=0.3, linestyles='solid'
                   )
        
        ################################ plot local theta positive anomaly
        orig_simulation = xr.open_dataset(orig_simulation_f[i])
        pres = orig_simulation.PS[0, 80:920, 80:920].values
        tem2m = orig_simulation.T_2M[0, 80:920, 80:920].values
        theta = tem2m * (p0sl/pres)**(r/cp)
        find_plot_maxmin_points(
            lon, lat, data=theta, ax=ax, extrema='max', nsize=50,
            pointcolor='b', pointsize=1)
        pdf.savefig(fig)
        plt.close('all')
        print(str(i) + '/' + str(len(time)))


identified_transformed_rvor_2010.close()

'''
# rvor100 = rvor_100m.relative_vorticity[i, 80:920, 80:920].values * 10**4
# rvor10 = rvor_10m.relative_vorticity[i, 80:920, 80:920].values * 10**4
'''

# endregion
# =============================================================================


# =============================================================================
# region display 100m and 10m rvor in 201008 and 201002 together

istart = np.where(time == np.datetime64('2010-08-03T20:00:00.000000000'))[0][0]
ifinal = np.where(time == np.datetime64('2010-08-09T08:00:00.000000000'))[0][0]
outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_0.2 Identified_rvor__theta_anomaly__100m_10m_rvor_20100803_09.mp4'
# istart = np.where(time == np.datetime64('2010-02-14T00:00:00.000000000'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-02-22T00:00:00.000000000'))[0][0]
# outputfile = 'figures/09_decades_vortex/09_03_require_theta_anomaly/9_3_0.3 Identified_rvor__theta_anomaly__100m_10m_rvor_20100214_21.mp4'

nrow = 1
ncol = 2
fig = plt.figure(figsize=np.array([8.8*ncol, 9.3 * nrow]) / 2.54, dpi = 600)
gs = fig.add_gridspec(nrow, ncol, wspace=0.05)
axs = gs.subplots(subplot_kw={'projection': transform}, sharey=True)

ims = []
for i in np.arange(istart, ifinal):  # np.arange(istart, istart + 1):  #
    plt_rvor = [0, 0]
    plt_text = [0, 0] # time
    # plt_title = [0, 0]  # title
    for j in range(ncol):
        if (j == 0):
            ################################ plot 100m rvor
            rvor = rvor_100m.relative_vorticity[i, 80:920, 80:920].values*10**4
        else:
            ################################ plot 10m rvor
            rvor = rvor_10m.relative_vorticity[i, 80:920, 80:920].values*10**4
        plt_rvor[j] = axs[j].pcolormesh(
            lon, lat, rvor, cmap=rvor_cmp,
            norm=BoundaryNorm(
                rvor_level, ncolors=rvor_cmp.N, clip=False),
            transform=transform, zorder=-2, rasterized=True)
        plt_text[j] = axs[j].text(
            -23, 34, str(time[i])[0:10] + ' ' + str(time[i])[11:13] + ':00 UTC')
        axs[j].set_extent(extent1km_lb, crs=transform)
    
    ims.append(plt_rvor + plt_text)
    print(str(i) + '/' + str(ifinal-1))

plt_gridline = [0, 0]
for j in range(ncol):
    # add borders, gridlines, title and set extent
    axs[j].add_feature(coastline, lw=0.25)
    axs[j].add_feature(borders, lw=0.25)
    plt_gridline[j] = axs[j].gridlines(
        crs=transform, linewidth=0.25,
        color='gray', alpha=0.5, linestyle='--')
    plt_gridline[j].xlocator = mticker.FixedLocator(ticklabel1km_lb[0])
    plt_gridline[j].ylocator = mticker.FixedLocator(ticklabel1km_lb[2])
    axs[j].set_xticks(ticklabel1km_lb[0])
    axs[j].set_xticklabels(ticklabel1km_lb[1])
    axs[j].set_yticks(ticklabel1km_lb[2])
    axs[j].set_yticklabels(ticklabel1km_lb[3])
    axs[j].contourf(
        lon, lat, model_topo_mask,
        colors='white', levels=np.array([0.5, 1.5]))
    scale_bar(axs[j], bars=2, length=200, location=(0.02, 0.015),
              barheight=20, linewidth=0.15, col='black', middle_label=False)

cbar = fig.colorbar(
    plt_rvor[0], ax = axs, orientation="horizontal",  pad=0.1,
    fraction=0.09, shrink=0.5, aspect=25, anchor = (0.5, 0.5),
    ticks=rvor_ticks, extend='both')
cbar.ax.set_xlabel(
    "100-meter (left) and 10-meter (right) relative vorticity [$10^{-4}\;s^{-1}$]")
# fig.tight_layout()
fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.99)
ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
ani.save(
    outputfile,
    progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'),)

# endregion
# =============================================================================




