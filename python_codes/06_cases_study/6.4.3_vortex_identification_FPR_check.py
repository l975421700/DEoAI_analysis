

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

from DEoAI_analysis.module.vortex_namelist import(
    correctly_identified201011,
    correctly_identified201011_sd
)

# endregion
# =============================================================================


smaller_domain = True
iyear = 4
# years[iyear]
# =============================================================================
# region import data

################################ 100m vorticity
rvor_100m_f = np.array(sorted(glob.glob(
    'scratch/rvorticity/relative_vorticity_1km_1h_100m/relative_vorticity_1km_1h_100m20' + years[iyear] + '*.nc')))
rvor_100m = xr.open_mfdataset(
    rvor_100m_f, concat_dim="time", data_vars='minimal',
    coords='minimal', compat='override', chunks={'time': 1})
time = rvor_100m.time.values
lon = rvor_100m.lon[80:920, 80:920].values
lat = rvor_100m.lat[80:920, 80:920].values

################################ wind data
# wind_earth_1h_100m = xr.open_dataset(
#     'scratch/wind_earth/wind_earth_1h_100m_strength_direction/wind_earth_1h_100m_strength_direction_20' + years[iyear] + '.nc')
# wind_100m_f = np.array(sorted(glob.glob(
#     'scratch/wind_earth/wind_earth_1h_100m/wind_earth_1h_100m20' + years[iyear] + '*.nc')))
# wind_100m = xr.open_mfdataset(
#     wind_100m_f, concat_dim="time", data_vars='minimal',
#     coords='minimal', compat='override', chunks={'time': 1})

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

previous_count = np.concatenate((np.array([0]), hourly_vortex_count[:-1]))
next_count = np.concatenate((hourly_vortex_count[1:], np.array([0]), ))
isolated_hour = np.vstack(((hourly_vortex_count > 0), (previous_count == 0),
                           (next_count == 0))).all(axis=0)
filtered_hourly_vortex_count = hourly_vortex_count.copy()
filtered_hourly_vortex_count[isolated_hour] = 0

################################ original simulation to calculate surface theta
orig_simulation_f = np.array(sorted(
    glob.glob('/store/c2sm/pr04/jvergara/RUNS_IN_SCRATCH/MAC1/lm_f/1h/lffd20' +
              years[iyear] + '*[0-9].nc')
))

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
        criteria1 = True
        if criteria1:
            ax.scatter(
                lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                s=pointsize, c=pointcolor,
            )

# endregion
# =============================================================================


# months[imonth]
imonth = 10
istart = np.where(time == np.datetime64('2010-11-01T00:00'))[0][0]
ifinal = np.where(time == np.datetime64('2010-12-01T00:00'))[0][0]

# imonth = 1 # Feb
# istart = np.where(time == np.datetime64('2010-02-17T00:00'))[0][0]
# ifinal = np.where(time == np.datetime64('2010-02-18T23:00'))[0][0]

# =============================================================================
# region plot reidentified rvor

if smaller_domain:
    outputfile = 'figures/09_decades_vortex/09_06_vortex_fpr_check/' + \
        '9_6_1 20' + years[iyear] + '_' + months[imonth] + ' ' + \
        'reidentified transformed rvor and local theta anomaly_sd.pdf'
else:
    outputfile = 'figures/09_decades_vortex/09_06_vortex_fpr_check/' + \
        '9_6_0 20' + years[iyear] + '_' + months[imonth] + ' ' + \
        'reidentified transformed rvor and local theta anomaly.pdf'

# time[[istart, ifinal]]
# orig_simulation_f[[istart, ifinal - 1]]

with PdfPages(outputfile) as pdf:
    for i in np.arange(istart, ifinal): # np.arange(istart, istart + 10):  #
        # i = istart
        
        ################################ extract vorticity, wind
        rvor = rvor_100m.relative_vorticity[i, 80:920, 80:920].values * 10**4
        
        ################################ plot rvor
        fig, ax = framework_plot1(
            "1km_lb",
            plot_vorticity=True,
            xlabel="Relative vorticity [$10^{-4}\;s^{-1}$]",
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
        is_vortex = experiment.vortex_info.cols.is_vortex[i]
        vortex_indices = \
            np.ma.array(experiment.vortex_info.cols.vortex_indices[i])
        vortex_indices[is_vortex == 0] = np.ma.masked
        vortex_count = experiment.vortex_info.cols.vortex_count[i]
        
        if (not isolated_hour[i]):
            for j in range(vortex_count):
                # j = 0
                ################################ vortex detailed info
                vortex_de_r = [row[:]
                               for row in experiment.vortex_de_info.where(
                               '(time_index == ' + str(i) +
                               ') & (vortex_index == ' + str(j) + ')')][0]
                vortex_de_cols = experiment.vortex_de_info.colnames
                center_lat = vortex_de_r[vortex_de_cols.index('center_lat')]
                center_lon = vortex_de_r[vortex_de_cols.index('center_lon')]
                vortex_radius = vortex_de_r[vortex_de_cols.index('radius')]
                
                ################################ plot vortex mean winds
                vortex_wind_u = vortex_de_r[vortex_de_cols.index('mean_wind_u')]
                vortex_wind_v = vortex_de_r[vortex_de_cols.index('mean_wind_v')]
                ax.quiver(center_lon, center_lat, vortex_wind_u, vortex_wind_v,
                          rasterized=True)
                
                ################################ plot a nominal circle
                vortex_circle = plt.Circle(
                    (center_lon, center_lat), vortex_radius/1.1*0.01,
                    edgecolor='lime', facecolor='None', lw=0.3, zorder=2)
                ax.add_artist(vortex_circle)
                
                ################################ plot text
                ax.text(
                    center_lon, center_lat,
                    str(j) + ':' +
                    str(int(np.pi * vortex_radius ** 2)) + ':' +
                    str(int(vortex_de_r[vortex_de_cols.index('angle')])) + ':' +
                    str(round(vortex_de_r[vortex_de_cols.index(
                        'distance2radius')], 1)),
                    color='m', size=6, fontweight='normal')
        
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
        print(str(i) + '/' + str(ifinal))


identified_rvor.close()


# endregion
# =============================================================================


# =============================================================================
# region check FPR

total_number = 0
time201011 = np.arange('2010-11-01T00', '2010-12-01T00', dtype='datetime64[h]')
for i in range(len(correctly_identified201011_sd)):
    total_number += len(
        correctly_identified201011_sd[str(time201011[i])[0:13]])

total_number / np.sum(filtered_hourly_vortex_count[istart:ifinal])
# 0.788, 0.212, 1/15.7h, 1/44.0
# sd: 0.859, 0.141, 199, 171, 1/25.7h
# endregion
# =============================================================================



